# Adlatus Chatbot 🤖  

A Retrieval-Augmented Generation (RAG) chatbot for [Adlatus Zürich](https://adlatus-zh.ch), built with **FastAPI**, **FAISS**, and **OpenAI’s Responses API**.  
It can answer questions based on scraped website pages, PDF documents, and structured contacts data.

---

## 🚀 Features

- **FastAPI Backend** – lightweight REST API with `/ask` endpoint  
- **RAG Pipeline** – combines OpenAI models with local context retrieval  
- **FAISS Index** – efficient vector search for knowledge retrieval  
- **Multi-source Knowledge** – supports website content, PDF documents, and contact data  
- **Deploy Anywhere** – works locally or on platforms like [Render](https://render.com)  

---

## 📂 Project Structure

adlatus-chatbot/
│
├── adlatus_rag/                # Main project folder
│   ├── scraper.py              # Scrape website pages
│   ├── ingest.py               # Extract, clean & chunk text
│   ├── chatbot.py              # Chatbot RAG logic
│   ├── utils/
│   │   └── text_cleaner.py     # Text preprocessing helpers
│   │
│   └── data/
│       ├── raw/                # Raw scraped HTML & PDFs
│       ├── processed/          # Processed JSON + contacts
│       └── index/              # FAISS vector index
│
├── app.py                      # FastAPI entrypoint (API server)
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (see below)
└── README.md                   # Project documentation
