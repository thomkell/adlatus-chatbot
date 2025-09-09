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
`````
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
`````

---
## Wordpress implemntation

`````
<style>
  #chatbot-container {
    position: fixed;
    bottom: 24px;
    right: 24px;
    width: 320px;
    max-height: 500px;
    border: 1px solid #ccc;
    border-radius: 12px;
    background: white;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    font-family: sans-serif;
    display: none;
    flex-direction: column;
    overflow: hidden;
    z-index: 9999;
  }

  #chatbot-header {
    background: #003366;
    color: white;
    padding: 12px;
    font-weight: bold;
  }

  #chatbot-messages {
    flex: 1;
    padding: 10px;
    overflow-y: auto;
    font-size: 14px;
  }

  #chatbot-messages div {
    margin-bottom: 10px;
  }

  .user-message {
    text-align: right;
    color: #003366;
  }

  .bot-message {
    text-align: left;
    color: #111;
  }

  #chatbot-input {
    display: flex;
    border-top: 1px solid #ccc;
  }

  #chatbot-input input {
    flex: 1;
    padding: 10px;
    border: none;
    font-size: 14px;
  }

  #chatbot-input button {
    background: #003366;
    color: white;
    border: none;
    padding: 0 16px;
    cursor: pointer;
  }

  #chatbot-toggle {
    position: fixed;
    bottom: 24px;
    right: 24px;
    width: 56px;
    height: 56px;
    background: #003366;
    border-radius: 50%;
    color: white;
    font-size: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    z-index: 9999;
  }
</style>

<div id="chatbot-toggle">💬</div>

<div id="chatbot-container">
  <div id="chatbot-header">Adlatus</div>
  <div id="chatbot-messages">
    <div class="bot-message">Hallo! Ich bin Adlatus. Wie kann ich dir helfen?</div>
  </div>
  <div id="chatbot-input">
    <input type="text" id="chatbot-question" placeholder="Frage eingeben..." />
    <button onclick="sendToAdlatus()">➤</button>
  </div>
</div>

<script>
  const toggle = document.getElementById('chatbot-toggle');
  const container = document.getElementById('chatbot-container');
  const messages = document.getElementById('chatbot-messages');
  const input = document.getElementById('chatbot-question');

  toggle.onclick = () => {
    container.style.display = container.style.display === 'flex' ? 'none' : 'flex';
    container.style.flexDirection = 'column';
  };

  async function sendToAdlatus() {
    const question = input.value.trim();
    if (!question) return;

    messages.innerHTML += `<div class="user-message">${question}</div>`;
    input.value = '';

    try {
      const res = await fetch('https://adlatus-chatbot.onrender.com/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: question })
      });

      const data = await res.json();

      if (data.type === 'contact' && data.contact) {
        const c = data.contact;
        messages.innerHTML += `<div class="bot-message">
          <strong>${c.name || 'Kontakt'}</strong><br>
          ${c.email ? '📧 ' + c.email + '<br>' : ''}
          ${c.phone ? '📞 ' + c.phone + '<br>' : ''}
          ${c.location ? '📍 ' + c.location + '<br>' : ''}
          ${c.profile_url ? '<a href="'+c.profile_url+'" target="_blank">Profil öffnen ↗</a>' : ''}
        </div>`;
      } else if (data.type === 'answer') {
        messages.innerHTML += `<div class="bot-message">${data.answer}</div>`;
      } else {
        messages.innerHTML += `<div class="bot-message">Entschuldigung, keine Antwort gefunden.</div>`;
      }

    } catch (err) {
      messages.innerHTML += `<div class="bot-message">Fehler beim Verbinden mit dem Server.</div>`;
    }

    messages.scrollTop = messages.scrollHeight;
  }

  // ✅ Send on Enter key
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault(); // prevents accidental line breaks
      sendToAdlatus();
    }
  });
</script>
`````
