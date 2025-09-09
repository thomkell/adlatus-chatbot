# app.py
import os, json, re
import numpy as np
import pandas as pd
from pathlib import Path
import faiss
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# ----- config -----
load_dotenv()
CONTACTS_PATH = os.getenv("CONTACTS_PATH", "data/processed/contacts.json")
INDEX_DIR     = os.getenv("INDEX_DIR", "data/index")
EMBED_MODEL   = os.getenv("EMBED_MODEL", "text-embedding-3-small")
GEN_MODEL     = os.getenv("GEN_MODEL", "gpt-4.1-mini")

# --- tiny auto-detection for repo layouts (uses env if set; otherwise falls back) ---
if not os.path.exists(CONTACTS_PATH):
    alt_contacts = "adlatus_rag/data/processed/contacts.json"
    if os.path.exists(alt_contacts):
        CONTACTS_PATH = alt_contacts

idx_candidate = os.path.join(INDEX_DIR, "faiss.index")
if not os.path.exists(idx_candidate):
    alt_index_dir = "adlatus_rag/data/index"
    if os.path.exists(os.path.join(alt_index_dir, "faiss.index")):
        INDEX_DIR = alt_index_dir

client = OpenAI()

app = FastAPI(title="Adlatus RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- load data (lazy) -----
CONTACTS = None
FAISS = None
META = None

def load_contacts():
    global CONTACTS
    if CONTACTS is None:
        if CONTACTS_PATH and os.path.exists(CONTACTS_PATH):
            with open(CONTACTS_PATH, "r", encoding="utf-8") as f:
                CONTACTS = json.load(f)
        else:
            CONTACTS = []
    return CONTACTS

def load_index():
    """Load FAISS + metadata if present. Return (None, None) if not."""
    global FAISS, META
    if FAISS is None or META is None:
        idx_path = os.path.join(INDEX_DIR, "faiss.index")
        # Graceful: if missing, report "not loaded" instead of crashing
        if not os.path.exists(idx_path):
            return None, None
        FAISS = faiss.read_index(idx_path)
        p_parq = os.path.join(INDEX_DIR, "metadata.parquet")
        p_csv  = os.path.join(INDEX_DIR, "metadata.csv")
        if os.path.exists(p_parq):
            META = pd.read_parquet(p_parq)
        elif os.path.exists(p_csv):
            META = pd.read_csv(p_csv)
        else:
            # No metadata available → treat as not loaded
            return None, None
    return FAISS, META

# ----- embeddings + retrieval -----
def embed(text: str) -> np.ndarray:
    e = client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding
    v = np.array(e, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(v)
    return v

def retrieve(query: str, k: int = 6) -> pd.DataFrame:
    fa, meta = load_index()
    if fa is None or meta is None:
        # No index → empty result; caller handles message
        return pd.DataFrame(columns=["title","url","text","score"])
    v = embed(query)
    D, I = fa.search(v, k)
    return meta.iloc[I[0]].assign(score=D[0]).reset_index(drop=True)

# ----- contact matching (same logic as CLI bot) -----
STOP_DE = {"wer","ist","bin","bist","sind","seid","für","fuer","der","die","das","den","dem","des",
           "ein","eine","einen","und","oder","mit","im","in","am","an","zu","zum","zur","vom","von",
           "auf","aus","auch","bei","ohne","um","welcher","welche","welches","was","wie","wo","wann",
           "warum","wieso","bitte","thema","zuständig","zustandig"}
GENERIC_EMAILS = {"info","kontakt","contact","office","support","hello","service","mail","team","adlatus-zurich"}
CONTACT_INTENT = {"wer","ansprechpartner","ansprechperson","kontakt","email","e-mail","telefon",
                  "zuständig","zustandig","berater","fachmann","experte","ansprechstelle"}

def is_contact_intent(q: str) -> bool:
    q = q.lower()
    return any(w in q for w in CONTACT_INTENT)

def _normalize(s: str) -> str:
    if not s: return ""
    s = s.lower()
    return s.replace("ä","ae").replace("ö","oe").replace("ü","ue").replace("ß","ss")

def _tokens(s: str): return re.findall(r"[a-z0-9]{2,}", _normalize(s))
def _content_tokens(s: str): return {t for t in _tokens(s) if t not in STOP_DE}

def _competency_tokens(c: dict):
    comp_text = " ".join((c.get("competencies") or []))
    extra = " ".join([c.get("name",""), c.get("title","")])
    return _content_tokens(comp_text) | _content_tokens(extra)

def _email_localpart(email: Optional[str]) -> Optional[str]:
    if not email or "@" not in email: return None
    return email.split("@",1)[0].lower()

def score_contact(query: str, c: dict):
    qtok = _content_tokens(query)
    ctok = _competency_tokens(c)
    if not ctok: return (-1e9, 0)
    overlap = len(qtok & ctok)
    jaccard = overlap / max(1, len(qtok | ctok))
    score = overlap + 2.0 * jaccard
    if c.get("email"): score += 0.4
    if c.get("phone"): score += 0.2
    if _email_localpart(c.get("email")) in GENERIC_EMAILS: score -= 0.6
    return (score, overlap)

def pick_best_contact(query: str):
    contacts = load_contacts()
    if not contacts: return None
    scored = [(lambda t: (t[0], t[1], c)) (score_contact(query, c)) for c in contacts]
    scored = [t for t in scored if t[1] >= 1]  # need ≥1 topical overlap
    if not scored: return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][2]

def format_contact(c: dict) -> dict:
    return {
        "name": c.get("name"),
        "email": c.get("email"),
        "phone": c.get("phone"),
        "location": c.get("location"),
        "competencies": c.get("competencies", [])[:10],
        "profile_url": c.get("profile_url"),
    }

# ----- LLM answer from PDFs -----
SYSTEM = (
    "You are Adlatus-ZH’s assistant. Always prioritize the provided context when answering. "
    "If the information is not in the context, you may use your general knowledge about Adlatus-ZH "
    "to provide an accurate answer. If you are still unsure, say you don't know and suggest checking "
    "the official Adlatus-ZH homepage or contacting them directly."
)


def answer_from_pdfs(query: str, k: int = 6) -> str:
    docs = retrieve(query, k=k)
    if docs.empty:
        return ("Ich habe dafür aktuell keinen konfigurierten Kontext. "
                "Bitte stelle sicher, dass FAISS-Index + Metadaten vorhanden sind "
                f"({INDEX_DIR}) oder setze INDEX_DIR/CONTACTS_PATH korrekt.")
    context = "\n\n".join(
        f"[{i+1}] {row.title} ({row.url})\n{row.text}"
        for i, row in docs.iterrows()
    )
    prompt = (f"Context documents:\n{context}\n\nUser question: {query}\n\n"
              f"Instructions: Cite sources inline like [1],[2] by their numbers.")
    resp = client.responses.create(
        model=GEN_MODEL,
        input=[{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],
    )
    return resp.output_text

# ----- API schema -----
class AskIn(BaseModel):
    query: str
    k: Optional[int] = 6

@app.get("/health")
def health():
    # minimal sanity check
    n_contacts = len(load_contacts() or [])
    try:
        fa, meta = load_index()
        n_index = 0 if fa is None else fa.ntotal
        meta_loaded = bool(meta is not None)
    except Exception:
        n_index = 0
        meta_loaded = False
    return {
        "ok": True,
        "contacts": n_contacts,
        "index": n_index,
        "meta_loaded": meta_loaded,
        "paths": {"contacts_path": CONTACTS_PATH, "index_dir": INDEX_DIR},
        "faiss_exists": os.path.exists(os.path.join(INDEX_DIR, "faiss.index")),
        "meta_parquet_exists": os.path.exists(os.path.join(INDEX_DIR, "metadata.parquet")),
        "meta_csv_exists": os.path.exists(os.path.join(INDEX_DIR, "metadata.csv")),
    }

@app.post("/ask")
def ask(inp: AskIn):
    q = inp.query.strip()
    # contacts only if clear intent
    if is_contact_intent(q):
        c = pick_best_contact(q)
        if c:
            return {"type":"contact", "contact": format_contact(c)}
        else:
            return {
                "type":"contact",
                "contact": None,
                "message":"Keine Kontakte geladen. Lege contacts.json an oder setze CONTACTS_PATH."
            }
    # otherwise PDFs
    ans = answer_from_pdfs(q, k=inp.k or 6)
    return {"type":"answer", "answer": ans}
