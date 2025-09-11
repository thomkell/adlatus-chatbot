import os, time, json
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
INDEX_DIR   = os.getenv("INDEX_DIR", "data/index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
GEN_MODEL   = os.getenv("GEN_MODEL", "gpt-4.1-mini")

client = OpenAI()

# ----- FAISS + metadata -----
FAISS = None
META  = None

def load_index():
    global FAISS, META
    if FAISS is None or META is None:
        idx_path = os.path.join(INDEX_DIR, "faiss.index")
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
            META = None
    return FAISS, META

def embed(text: str) -> np.ndarray:
    e = client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding
    v = np.array(e, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(v)
    return v

def retrieve(query: str, k: int = 6) -> pd.DataFrame:
    fa, meta = load_index()
    if fa is None or meta is None:
        return pd.DataFrame()
    v = embed(query)
    D, I = fa.search(v, k)
    return meta.iloc[I[0]].assign(score=D[0]).reset_index(drop=True)

# ----- FastAPI setup -----
app = FastAPI(title="Adlatus RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- session memory -----
SESSION_MEMORY = {}
SESSION_TTL = 1800   # 30 minutes

def cleanup_sessions():
    now = time.time()
    expired = [sid for sid, s in SESSION_MEMORY.items() if now - s["last_used"] > SESSION_TTL]
    for sid in expired:
        del SESSION_MEMORY[sid]

def init_session(session_id):
    cleanup_sessions()
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = {"history": [], "last_used": time.time()}
    return SESSION_MEMORY[session_id]

def add_to_history(session_id, role, content, max_len=10):
    s = init_session(session_id)
    s["history"].append((role, content))
    s["last_used"] = time.time()
    s["history"] = s["history"][-max_len:]

def get_history(session_id):
    s = init_session(session_id)
    return [{"role": r, "content": c} for r, c in s["history"]]

# ----- system prompt -----
SYSTEM = (
    "You are Adlatus-ZH’s assistant. Always prioritize the provided context when answering. "
    "When you use the context, cite sources inline like [1], [2] using the provided document numbers. "
    "If the information is not in the context, you may use your general knowledge about Adlatus-ZH, "
    "but keep the answer very brief (maximum 2 sentences) and do not add citations. "
    "If you are still unsure, say you don't know and suggest checking the official Adlatus-ZH homepage "
    "or contacting them directly. Answer in the user's language."
)

# ----- API schema -----
class AskIn(BaseModel):
    query: str
    k: Optional[int] = 6
    session_id: Optional[str] = "default"

@app.get("/health")
def health():
    try:
        fa, meta = load_index()
        n_index = 0 if fa is None else fa.ntotal
        meta_loaded = bool(meta is not None)
    except Exception:
        n_index = 0
        meta_loaded = False
    return {
        "ok": True,
        "index": n_index,
        "meta_loaded": meta_loaded,
        "paths": {"index_dir": INDEX_DIR},
        "faiss_exists": os.path.exists(os.path.join(INDEX_DIR, "faiss.index")),
    }

@app.post("/ask")
def ask(inp: AskIn):
    q = inp.query.strip()
    session_id = inp.session_id or "default"
    session = init_session(session_id)

    # ---- retrieve from FAISS ----
    docs = retrieve(q, k=inp.k or 6)
    if docs.empty:
        msg = "Keine Daten gefunden. Bitte prüfen Sie die Index-Dateien."
        return {"type": "error", "message": msg, "session_id": session_id}

    # ---- contacts ----
    contacts = docs[docs["source"] == "contact"]
    if not contacts.empty:
        results = []
        for _, row in contacts.iterrows():
            results.append({
                "name": row.get("name"),
                "email": row.get("email"),
                "phone": row.get("phone"),
                "location": row.get("location"),
                "profile_url": row.get("url"),
                "competencies": row.get("text"),
            })
        add_to_history(session_id, "user", q)
        add_to_history(session_id, "assistant", f"Gefundene Kontakte: {[c['name'] for c in results]}")
        return {"type": "contacts", "contacts": results, "session_id": session_id}

    # ---- otherwise normal RAG (PDFs etc.) ----
    context = "\n\n".join(
        f"[{i+1}] {row.title} ({row.url})\n{row.text}"
        for i, row in docs.iterrows() if row["source"] == "pdf"
    )

    history = get_history(session_id)
    messages = [{"role": "system", "content": SYSTEM}]
    messages.extend(history)
    messages.append({
        "role": "user",
        "content": f"Context documents:\n{context}\n\nUser question: {q}"
    })

    resp = client.responses.create(model=GEN_MODEL, input=messages)
    answer = resp.output_text.strip()

    add_to_history(session_id, "user", q)
    add_to_history(session_id, "assistant", answer)

    return {"type": "answer", "answer": answer, "session_id": session_id}
