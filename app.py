# app.py
import os, json, re, time
import numpy as np
import pandas as pd
import faiss
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import random


# =====================================================
# ----- CONFIGURATION -----
# =====================================================
load_dotenv()

# Paths and model names (can be overridden via .env)
CONTACTS_PATH = os.getenv("CONTACTS_PATH", "data/processed/contacts.json")
INDEX_DIR     = os.getenv("INDEX_DIR", "data/index")
EMBED_MODEL   = os.getenv("EMBED_MODEL", "text-embedding-3-small")
GEN_MODEL     = os.getenv("GEN_MODEL", "gpt-4.1-mini")

# Try to auto-detect correct repo layout if paths don't exist
if not os.path.exists(CONTACTS_PATH):
    alt_contacts = "adlatus_rag/data/processed/contacts.json"
    if os.path.exists(alt_contacts):
        CONTACTS_PATH = alt_contacts

idx_candidate = os.path.join(INDEX_DIR, "faiss.index")
if not os.path.exists(idx_candidate):
    alt_index_dir = "adlatus_rag/data/index"
    if os.path.exists(os.path.join(alt_index_dir, "faiss.index")):
        INDEX_DIR = alt_index_dir

# OpenAI client
client = OpenAI()

# FastAPI app with CORS enabled
app = FastAPI(title="Adlatus RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (relax for prod)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# ----- SESSION MEMORY (short-term conversation memory) -----
# =====================================================
SESSION_MEMORY = {}  # {session_id: {"history": [(role,msg),...], "last_contacts": [...], "last_used": ts}}
SESSION_TTL = 1800   # 30 minutes of inactivity -> session expired

def cleanup_sessions():
    """Remove expired sessions from memory."""
    now = time.time()
    expired = [sid for sid, s in SESSION_MEMORY.items() if now - s["last_used"] > SESSION_TTL]
    for sid in expired:
        del SESSION_MEMORY[sid]

def init_session(session_id: str):
    """Initialize or refresh a session."""
    cleanup_sessions()
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = {"history": [], "last_contacts": [], "last_used": time.time()}
    return SESSION_MEMORY[session_id]

def add_to_history(session_id, role, content, max_len=10):
    """Append message to session history, keep last N messages."""
    s = init_session(session_id)
    s["history"].append((role, content))
    s["last_used"] = time.time()
    s["history"] = s["history"][-max_len:]

def get_history(session_id):
    """Return session history in OpenAI-friendly format."""
    return [{"role": r, "content": c} for r, c in init_session(session_id)["history"]]


# =====================================================
# ----- TEXT NORMALIZATION -----
# =====================================================
def normalize_query(text: str) -> str:
    """Lowercase query and normalize umlauts/ß for consistent matching."""
    return (
        text.lower()
        .replace("ä","ae").replace("ö","oe").replace("ü","ue").replace("ß","ss")
    )


# =====================================================
# ----- DATA LOADING (Contacts + FAISS index) -----
# =====================================================
CONTACTS = None
FAISS = None
META = None

def load_contacts():
    """Load contacts JSON file into memory (cached)."""
    global CONTACTS
    if CONTACTS is None:
        if os.path.exists(CONTACTS_PATH):
            with open(CONTACTS_PATH, "r", encoding="utf-8") as f:
                CONTACTS = json.load(f)
        else:
            CONTACTS = []
    return CONTACTS

def load_index():
    """Load FAISS index and metadata (cached)."""
    global FAISS, META
    if FAISS is None or META is None:
        idx_path = os.path.join(INDEX_DIR, "faiss.index")
        if not os.path.exists(idx_path):
            return None, None
        FAISS = faiss.read_index(idx_path)

        # Try parquet first (smaller, faster), fallback to CSV
        p_parq = os.path.join(INDEX_DIR, "metadata.parquet")
        p_csv  = os.path.join(INDEX_DIR, "metadata.csv")
        if os.path.exists(p_parq):
            META = pd.read_parquet(p_parq)
        elif os.path.exists(p_csv):
            META = pd.read_csv(p_csv)
        else:
            return None, None
    return FAISS, META


# =====================================================
# ----- EMBEDDINGS + RETRIEVAL -----
# =====================================================
def embed(text: str) -> np.ndarray:
    """Create OpenAI embedding vector for a given text (normalized L2)."""
    e = client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding
    v = np.array(e, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(v)
    return v

def retrieve(query: str, k: int = 6) -> pd.DataFrame:
    """Retrieve top-k relevant docs from FAISS index for a query."""
    fa, meta = load_index()
    if fa is None or meta is None:
        return pd.DataFrame(columns=["title","url","text","score"])
    v = embed(query)
    D, I = fa.search(v, k)
    return meta.iloc[I[0]].assign(score=D[0]).reset_index(drop=True)


# =====================================================
# ----- CONTACT MATCHING -----
# =====================================================
# Stopwords (ignored when scoring)
STOP_DE = {"wer","ist","bin","bist","sind","seid","für","fuer","der","die","das","den","dem","des",
           "ein","eine","einen","und","oder","mit","im","in","am","an","zu","zum","zur","vom","von",
           "auf","aus","auch","bei","ohne","um","welcher","welche","welches","was","wie","wo","wann",
           "warum","wieso","bitte","thema","zuständig","zustandig"}

# Ignore these generic emails
GENERIC_EMAILS = {"info","kontakt","contact","office","support","hello","service","mail","team","adlatus-zurich"}

# Keywords that indicate user is asking for a contact
CONTACT_INTENT = {"wer","ansprechpartner","ansprechperson","kontakt","email","telefon","zuständig","zustandig","berater"}

def is_contact_intent(q: str) -> bool:
    """Check if query is asking about contacts (via keyword detection)."""
    q = normalize_query(q)
    return any(w in q for w in CONTACT_INTENT)

# ---- tokenization helpers ----
def _normalize(s: str) -> str:
    return (s or "").lower().replace("ä","ae").replace("ö","oe").replace("ü","ue").replace("ß","ss")

def _tokens(s: str): return re.findall(r"[a-z0-9]{2,}", _normalize(s))
def _content_tokens(s: str): return {t for t in _tokens(s) if t not in STOP_DE}

def _competency_tokens(c: dict):
    """Extract tokens from contact’s competencies + name + title."""
    comp_text = " ".join(c.get("competencies") or [])
    extra = " ".join([c.get("name",""), c.get("title","")])
    return _content_tokens(comp_text) | _content_tokens(extra)

def _email_localpart(email: Optional[str]) -> Optional[str]:
    """Get local part of email (before @)."""
    if email and "@" in email:
        return email.split("@",1)[0].lower()
    return None

def score_contact(query: str, c: dict):
    """Score contact against query using overlap + heuristics."""
    qtok = _content_tokens(query)
    ctok = _competency_tokens(c)
    if not ctok: return (-1e9, 0)  # ignore empty contacts

    overlap = len(qtok & ctok)
    jaccard = overlap / max(1, len(qtok | ctok))
    score = overlap + 2*jaccard  # weight overlap + similarity

    # Heuristics to reward detailed contacts
    if c.get("email"): score += 0.4
    if c.get("phone"): score += 0.2
    if _email_localpart(c.get("email")) in GENERIC_EMAILS: score -= 0.6

    return score, overlap

def format_contact(c: dict) -> dict:
    """Return contact info in safe, consistent format."""
    return {
        "name": c.get("name"),
        "email": c.get("email"),
        "phone": c.get("phone"),
        "location": c.get("location"),
        "competencies": c.get("competencies", [])[:10],
        "profile_url": c.get("profile_url"),
    }

def pick_matching_contacts(query: str, max_results: int = 2, pool_size: int = 5):
    """
    Pick up to `max_results` contacts matching query.
    - Score all contacts
    - Keep top `pool_size`
    - Randomly pick from pool (avoids always same contacts)
    """
    contacts = load_contacts()
    if not contacts:
        return []

    scored = []
    for c in contacts:
        sc, ov = score_contact(query, c)
        if ov >= 1:  # require at least 1 overlap token
            scored.append((sc, ov, c))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    pool = scored[:pool_size]

    # Random selection introduces variation
    chosen = random.sample(pool, min(max_results, len(pool)))

    return [format_contact(t[2]) for t in chosen]


# =====================================================
# ----- SYSTEM PROMPT -----
# =====================================================
SYSTEM = (
    "You are Adlatus-ZH’s assistant. Always prioritize the provided context when answering. "
    "When you use the context, cite sources inline like [1],[2]. "
    "If info not in context, keep the answer brief (≤2 sentences). "
    "If unsure, say you don't know and suggest checking the official Adlatus-ZH homepage. "
    "Answer in the user's language."
)


# =====================================================
# ----- API SCHEMA -----
# =====================================================
class AskIn(BaseModel):
    query: str
    k: Optional[int] = 6
    session_id: Optional[str] = "default"


# =====================================================
# ----- API ENDPOINTS -----
# =====================================================
@app.get("/health")
def health():
    """Health check endpoint: shows if contacts + index are loaded."""
    n_contacts = len(load_contacts() or [])
    try:
        fa, meta = load_index()
        n_index = fa.ntotal if fa else 0
        meta_loaded = meta is not None
    except Exception:
        n_index, meta_loaded = 0, False
    return {
        "ok": True,
        "contacts": n_contacts,
        "index": n_index,
        "meta_loaded": meta_loaded,
    }

@app.post("/ask")
def ask(inp: AskIn):
    """Main endpoint: decides whether to return contacts or RAG-based answer."""
    q = inp.query.strip()
    norm_q = normalize_query(q)
    session_id = inp.session_id or "default"
    session = init_session(session_id)

    # --- 1. Contact intent ---
    if is_contact_intent(norm_q):
        matches = pick_matching_contacts(norm_q, max_results=2)
        if matches:
            session["last_contacts"] = matches
            add_to_history(session_id, "user", q)
            return {"type": "contacts", "contacts": matches, "session_id": session_id}
        return {"type": "contacts", "contacts": [], "message": "Keine passenden Kontakte gefunden.", "session_id": session_id}

    # --- 2. General Q&A via RAG pipeline ---
    docs = retrieve(q, k=inp.k or 6)
    context = ""
    if not docs.empty:
        context = "\n\n".join(f"[{i+1}] {row.title} ({row.url})\n{row.text}" for i,row in docs.iterrows())

    messages = [{"role":"system","content":SYSTEM}] + get_history(session_id) + [
        {"role":"user","content": f"Context:\n{context}\n\nUser: {q}"}
    ]

    # Call OpenAI Responses API
    resp = client.responses.create(model=GEN_MODEL, input=messages)
    answer = resp.output_text.strip()

    # Update memory
    add_to_history(session_id, "user", q)
    add_to_history(session_id, "assistant", answer)

    return {"type": "answer", "answer": answer, "session_id": session_id}
