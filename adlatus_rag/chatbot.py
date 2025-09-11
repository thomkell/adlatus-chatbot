# chatbot.py — FAISS-only (contacts + PDFs)
import os
import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv
from openai import OpenAI

# ---------- setup ----------
load_dotenv()
client = OpenAI()

INDEX_DIR = "data/index"
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
GEN_MODEL   = os.getenv("GEN_MODEL", "gpt-4.1-mini")

# ---------- load FAISS + metadata ----------
faiss_index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
meta_path_parquet = os.path.join(INDEX_DIR, "metadata.parquet")
meta_path_csv     = os.path.join(INDEX_DIR, "metadata.csv")

if os.path.exists(meta_path_parquet):
    META = pd.read_parquet(meta_path_parquet)
elif os.path.exists(meta_path_csv):
    META = pd.read_csv(meta_path_csv)
else:
    raise FileNotFoundError("No metadata found (expected metadata.parquet or metadata.csv in data/index).")

# ---------- embeddings ----------
def embed(text: str) -> np.ndarray:
    e = client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding
    v = np.array(e, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(v)
    return v

# ---------- retrieval ----------
def retrieve(query: str, k: int = 6) -> pd.DataFrame:
    v = embed(query)
    D, I = faiss_index.search(v, k)
    hits = META.iloc[I[0]].assign(score=D[0]).reset_index(drop=True)
    return hits

# ---------- format outputs ----------
def format_contact(row: pd.Series) -> str:
    lines = []
    if row.get("name"):      lines.append(f"**Ansprechpartner:** {row['name']}")
    if row.get("email"):     lines.append(f"**E-Mail:** {row['email']}")
    if row.get("phone"):     lines.append(f"**Telefon:** {row['phone']}")
    if row.get("location"):  lines.append(f"**Ort:** {row['location']}")
    if row.get("text"):      lines.append("**Kernkompetenzen:** " + row["text"])
    if row.get("url"):       lines.append(f"**Profil:** {row['url']}")
    return "\n".join(lines)

SYSTEM = (
    "You are Adlatus-ZH’s assistant. Answer using ONLY the provided context. "
    "If the answer isn't present, say you don't know and suggest checking the Adlatus PDFs."
)

def generate_from_pdfs(query: str, k: int = 6) -> str:
    docs = retrieve(query, k=k)
    context = "\n\n".join(
        f"[{i+1}] {row.title} ({row.url})\n{row.text}"
        for i, row in docs.iterrows() if row["source"] == "pdf"
    )
    if not context:
        return "Keine passenden Informationen gefunden."
    prompt = (
        f"Context documents:\n{context}\n\n"
        f"User question: {query}\n\n"
        f"Instructions: Cite sources inline like [1],[2] by their numbers."
    )
    resp = client.responses.create(
        model=GEN_MODEL,
        input=[{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],
    )
    return resp.output_text

# ---------- main answer ----------
def answer(query: str) -> str:
    docs = retrieve(query, k=6)

    # 1) Kontakte zuerst
    contacts = docs[docs["source"] == "contact"]
    if not contacts.empty:
        row = contacts.iloc[0]  # best match
        return format_contact(row)

    # 2) Sonst PDFs
    return generate_from_pdfs(query, k=6)

# ---------- CLI ----------
if __name__ == "__main__":
    print("Adlatus Chatbot (FAISS-only: contacts + PDFs). Press Enter to exit.\n")
    while True:
        try:
            q = input("You › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        print("\nAdlatus › " + answer(q) + "\n")
