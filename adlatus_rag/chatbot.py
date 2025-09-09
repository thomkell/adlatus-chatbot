# chatbot.py — contacts first on intent (competencies-only), else PDFs via FAISS
import os, json, re
import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv
from openai import OpenAI

# ---------- setup ----------
load_dotenv()
client = OpenAI()

CONTACTS_PATH = "data/processed/contacts.json"
INDEX_DIR     = "data/index"

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
GEN_MODEL   = os.getenv("GEN_MODEL", "gpt-4.1-mini")

# ---------- load contacts ----------
def load_contacts(path=CONTACTS_PATH):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
CONTACTS = load_contacts()

# ---------- load FAISS + metadata (from PDFs) ----------
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

# ---------- retrieval over PDFs ----------
def retrieve(query: str, k: int = 6) -> pd.DataFrame:
    v = embed(query)
    D, I = faiss_index.search(v, k)
    hits = META.iloc[I[0]].assign(score=D[0]).reset_index(drop=True)
    return hits

# ---------- contact matching (competencies-only) ----------
STOP_DE = {
    "wer","ist","bin","bist","sind","seid","für","fuer","der","die","das","den","dem","des","ein","eine","einen",
    "und","oder","mit","im","in","am","an","zu","zum","zur","vom","von","auf","aus","auch","bei","ohne","um",
    "welcher","welche","welches","was","wie","wo","wann","warum","wieso","bitte","thema","zuständig","zustandig"
}
GENERIC_EMAILS = {"info","kontakt","contact","office","support","hello","service","mail","team","adlatus-zurich"}

CONTACT_INTENT = {
    "wer","ansprechpartner","ansprechperson","kontakt","email","e-mail","telefon",
    "zuständig","zustandig","berater","fachmann","experte","ansprechstelle"
}

def is_contact_intent(query: str) -> bool:
    q = query.lower()
    return any(w in q for w in CONTACT_INTENT)

def _normalize(s: str) -> str:
    if not s: return ""
    s = s.lower()
    return (s.replace("ä","ae").replace("ö","oe").replace("ü","ue").replace("ß","ss"))

def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]{2,}", _normalize(s))

def _content_tokens(s: str) -> set[str]:
    return {t for t in _tokens(s) if t not in STOP_DE}

def _competency_tokens(c: dict) -> set[str]:
    comp_text = " ".join((c.get("competencies") or []))
    extra = " ".join([c.get("name",""), c.get("title","")])
    return _content_tokens(comp_text) | _content_tokens(extra)

def _email_localpart(email: str | None) -> str | None:
    if not email or "@" not in email: return None
    return email.split("@",1)[0].lower()

def score_contact(query: str, c: dict) -> tuple[float,int]:
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
    """Return best contact if there is at least 1 topical overlap."""
    if not CONTACTS: return None
    scored = []
    for c in CONTACTS:
        sc, ov = score_contact(query, c)
        scored.append((sc, ov, c))
    # require at least 1 overlapping token with competencies
    scored = [t for t in scored if t[1] >= 1]
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][2]

def format_contact(c: dict) -> str:
    lines = []
    if c.get("name"):      lines.append(f"**Ansprechpartner:** {c['name']}")
    if c.get("email"):     lines.append(f"**E-Mail:** {c['email']}")
    if c.get("phone"):     lines.append(f"**Telefon:** {c['phone']}")
    if c.get("location"):  lines.append(f"**Ort:** {c['location']}")
    if c.get("competencies"):
        lines.append("**Kernkompetenzen:** " + ", ".join(c["competencies"][:10]))
    if c.get("profile_url"):
        lines.append(f"**Profil:** {c['profile_url']}")
    return "\n".join(lines)

# ---------- RAG generation over PDFs ----------
SYSTEM = (
    "You are Adlatus-ZH’s assistant. Answer using ONLY the provided context. "
    "If the answer isn't present, say you don't know and suggest reading the Adlatus PDFs."
)

def generate_from_pdfs(query: str, k: int = 6) -> str:
    docs = retrieve(query, k=k)
    context = "\n\n".join(
        f"[{i+1}] {row.title} ({row.url})\n{row.text}"
        for i, row in docs.iterrows()
    )
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
    # 1) Only try contacts if there is clear contact intent AND we actually have contacts
    if CONTACTS and is_contact_intent(query):
        best = pick_best_contact(query)
        if best:
            return format_contact(best)
    # 2) Otherwise: answer from PDFs (or if contacts unavailable/no match)
    return generate_from_pdfs(query, k=6)

# ---------- CLI ----------
if __name__ == "__main__":
    print("Adlatus Chatbot (contacts on intent; otherwise PDFs). Press Enter to exit.\n")
    while True:
        try:
            q = input("You › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        print("\nAdlatus › " + answer(q) + "\n")
