# app.py
import os, json, re
import numpy as np
import pandas as pd
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
        if os.path.exists(CONTACTS_PATH):
            with open(CONTACTS_PATH, "r", encoding="utf-8") as f:
                CONTACTS = json.load(f)
        else:
            CONTACTS = []
    return CONTACTS

def load_index():
    global FAISS, META
    if FAISS is None or META is None:
        FAISS = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
        p_parq = os.path.join(INDEX_DIR, "metadata.parquet")
        p_csv  = os.path.join(INDEX_DIR, "metadata.csv")
        if os.path.exists(p_parq):
            META = pd.read_parquet(p_parq)
        elif os.path.exists(p_csv):
            META = pd.read_csv(p_csv)
        else:
            raise FileNotFoundError("No metadata found in data/index")
    return FAISS, META

# ----- embeddings + retrieval -----
def embed(text: str) -> np.ndarray:
    e = client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding
    v = np.array(e, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(v)
    return v

def retrieve(query: str, k: int = 6) -> pd.DataFrame:
    fa, meta = load_index()
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
SYSTEM = ("You are Adlatus-ZH’s assistant. Answer using ONLY the provided context. "
          "If the answer isn't present, say you don't know and suggest reading the Adlatus PDFs.")

def answer_from_pdfs(query: str, k: int = 6) -> str:
    docs = retrieve(query, k=k)
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
        n_index = fa.ntotal
    except Exception:
        n_index = 0
    return {"ok": True, "contacts": n_contacts, "index": n_index}

@app.post("/ask")
def ask(inp: AskIn):
    q = inp.query.strip()
    # contacts only if clear intent
    if is_contact_intent(q):
        c = pick_best_contact(q)
        if c:
            return {"type":"contact", "contact": format_contact(c)}
    # otherwise PDFs
    ans = answer_from_pdfs(q, k=inp.k or 6)
    return {"type":"answer", "answer": ans}
