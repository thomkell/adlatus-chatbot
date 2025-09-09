# ingest.py  — PDFs only (clean & minimal)
import os, re, glob, argparse, pathlib
from dataclasses import dataclass, asdict
from typing import List, Dict

# ---------- PDF text extraction ----------
import fitz  # PyMuPDF

def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def extract_pdf_pages(pdf_path: str) -> List[Dict]:
    """Return list of {'page': int, 'text': str} for a PDF."""
    pages = []
    doc = fitz.open(pdf_path)
    try:
        for i in range(len(doc)):
            page = doc[i]
            try:
                text = page.get_text("text")
            except Exception:
                text = page.get_text()  # fallback
            text = _norm_ws(text)
            # light header/footer trim
            lines = [ln.strip() for ln in re.split(r"[\r\n]+", text) if ln.strip()]
            if len(lines) > 6:
                if len(lines[0]) < 40: lines = lines[1:]
                if len(lines[-1]) < 40: lines = lines[:-1]
            pages.append({"page": i + 1, "text": _norm_ws(" ".join(lines))})
    finally:
        doc.close()
    return pages

def pdf_title(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        title = (doc.metadata or {}).get("title") or os.path.basename(pdf_path)
        doc.close()
        return title
    except Exception:
        return os.path.basename(pdf_path)

# ---------- tokenization / chunking ----------
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:
    _enc = None

def count_tokens(s: str) -> int:
    if _enc is None:
        return max(1, len(s) // 4)  # rough fallback
    return len(_enc.encode(s))

def chunk_text(text: str, max_tokens=500, overlap_tokens=50) -> List[str]:
    if not text:
        return []
    if _enc is None:
        step = (max_tokens - overlap_tokens) * 4
        ov = overlap_tokens * 4
        chunks, i, n = [], 0, len(text)
        while i < n:
            chunks.append(text[i:min(n, i + step + ov)])
            i += step
        return chunks
    toks = _enc.encode(text)
    chunks, step = [], max_tokens - overlap_tokens
    for i in range(0, len(toks), step):
        piece = toks[i:i + max_tokens]
        chunks.append(_enc.decode(piece))
    return chunks

# ---------- embeddings (OpenAI) ----------
from dotenv import load_dotenv; load_dotenv()
from openai import OpenAI
client = OpenAI()

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

def embed_texts(texts: List[str]) -> List[List[float]]:
    out, B = [], 256
    for i in range(0, len(texts), B):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i+B])
        out.extend([d.embedding for d in resp.data])
    return out

# ---------- data model ----------
@dataclass
class Chunk:
    id: str
    url: str
    title: str
    text: str
    n_tokens: int
    source: str  # "pdf"

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Ingest PDFs → FAISS index")
    ap.add_argument("--pdf_dir", default="data/raw/pdfs", help="Folder with PDFs")
    ap.add_argument("--out_dir", default="data/processed", help="Where to write chunks.jsonl")
    ap.add_argument("--index_dir", default="data/index", help="Where to write faiss.index + metadata")
    ap.add_argument("--max_tokens", type=int, default=500)
    ap.add_argument("--overlap_tokens", type=int, default=50)
    args = ap.parse_args()

    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.index_dir).mkdir(parents=True, exist_ok=True)

    # 1) Collect PDF records
    pdf_paths = sorted(glob.glob(os.path.join(args.pdf_dir, "*.pdf")))
    if not pdf_paths:
        print(f"No PDFs found in {args.pdf_dir}. Nothing to ingest.")
        return

    records: List[Dict] = []
    for p in pdf_paths:
        title = pdf_title(p)
        for pg in extract_pdf_pages(p):
            records.append({
                "url": f"file://{os.path.abspath(p)}#page={pg['page']}",
                "title": f"{title} (Seite {pg['page']})",
                "text": pg["text"],
                "source": "pdf",
            })

    # 2) Chunk
    chunks: List[Chunk] = []
    for r in records:
        parts = chunk_text(r["text"], args.max_tokens, args.overlap_tokens)
        for j, p in enumerate(parts):
            cid = f"{r['url']}#chunk-{j:04d}"
            chunks.append(Chunk(
                id=cid, url=r["url"], title=r["title"],
                text=p, n_tokens=count_tokens(p), source="pdf"
            ))
    if not chunks:
        print("No chunks produced. Check your PDFs.")
        return

    # 3) Save chunks
    import json
    chunks_path = os.path.join(args.out_dir, "chunks.jsonl")
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    print(f"Wrote {len(chunks)} chunks -> {chunks_path}")

    # 4) Embed
    texts = [c.text for c in chunks]
    embs = embed_texts(texts)

    # 5) Build FAISS
    import numpy as np, faiss
    X = np.array(embs, dtype="float32")
    faiss.normalize_L2(X)  # cosine similarity via inner product
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    faiss.write_index(index, os.path.join(args.index_dir, "faiss.index"))

    # 6) Metadata (Parquet if available, else CSV)
    import pandas as pd
    meta_df = pd.DataFrame([asdict(c) for c in chunks])
    try:
        import pyarrow as _pa  # noqa: F401
        meta_df.to_parquet(os.path.join(args.index_dir, "metadata.parquet"))
        print(f"Index size: {index.ntotal} vectors; wrote metadata.parquet")
    except Exception as e:
        csv_p = os.path.join(args.index_dir, "metadata.csv")
        meta_df.to_csv(csv_p, index=False)
        print(f"Index size: {index.ntotal} vectors; parquet unavailable ({e}); wrote metadata.csv")

if __name__ == "__main__":
    main()
