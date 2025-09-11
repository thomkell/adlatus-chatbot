# Adlatus Chatbot 🤖  

A **Retrieval-Augmented Generation (RAG)** chatbot for [Adlatus Zürich](https://adlatus-zh.ch), built with **FastAPI**, **FAISS**, and **OpenAI’s Responses API**.  
It answers questions based on **processed data** (e.g. contacts, PDFs, prepared documents) using **FAISS vector search**.  
👉 Web scraping was used during initial data collection but is **not required for the current chatbot setup**.  

---

## 🚀 Features

- **FastAPI Backend** – lightweight REST API with `/ask` endpoint  
- **FAISS Index Only** – all answers are retrieved from a vector index of processed data  
- **Multi-source Knowledge** – supports contacts and PDFs (extendable to other structured sources)  
- **WordPress Integration** – chatbot widget that can be embedded in any WordPress site  
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
## ❓ What's next?

- 📂 **Additional Data Sources** – Add more PDFs, structured databases, or CRM integrations.


---
## 🌐 WordPress embedding

Paste this snippet into a Custom HTML block on your site:
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
    display: flex;
    justify-content: space-between;
    align-items: center;
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

  #chatbot-reset {
    font-size: 12px;
    background: transparent;
    border: none;
    color: #fff;
    cursor: pointer;
  }
</style>

<div id="chatbot-toggle">💬</div>

<div id="chatbot-container">
  <div id="chatbot-header">
    Adlatus
    <button id="chatbot-reset">Reset</button>
  </div>
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
  const resetBtn = document.getElementById('chatbot-reset');

  // 🔑 generate or reuse a session_id
  let sessionId = localStorage.getItem("adlatus_session_id");
  if (!sessionId) {
    sessionId = "sess-" + Math.random().toString(36).substr(2, 9);
    localStorage.setItem("adlatus_session_id", sessionId);
  }

  toggle.onclick = () => {
    container.style.display = container.style.display === 'flex' ? 'none' : 'flex';
    container.style.flexDirection = 'column';
  };

  // 🔄 reset session (start fresh conversation)
  resetBtn.onclick = () => {
    sessionId = "sess-" + Math.random().toString(36).substr(2, 9);
    localStorage.setItem("adlatus_session_id", sessionId);
    messages.innerHTML = `<div class="bot-message">Neue Sitzung gestartet. Hallo! Wie kann ich dir helfen?</div>`;
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
        body: JSON.stringify({ query: question, session_id: sessionId })
      });

      const data = await res.json();

      if (data.type === 'contacts' && data.contacts && data.contacts.length > 0) {
        data.contacts.forEach(c => {
          messages.innerHTML += `<div class="bot-message">
            <strong>${c.name || 'Kontakt'}</strong><br>
            ${c.email ? '📧 ' + c.email + '<br>' : ''}
            ${c.phone ? '📞 ' + c.phone + '<br>' : ''}
            ${c.location ? '📍 ' + c.location + '<br>' : ''}
            ${c.profile_url ? '<a href="'+c.profile_url+'" target="_blank">Profil öffnen ↗</a>' : ''}
          </div>`;
        });

      } else if (data.type === 'answer') {
        messages.innerHTML += `<div class="bot-message">${data.answer}</div>`;

      } else if (data.message) {
        messages.innerHTML += `<div class="bot-message">${data.message}</div>`;

      } else {
        messages.innerHTML += `<div class="bot-message">Entschuldigung, keine Antwort gefunden.</div>`;
      }

    } catch (err) {
      messages.innerHTML += `<div class="bot-message">⚠️ Der Chatbot ist momentan nicht erreichbar. Bitte versuche es später erneut.</div>`;
    }

    messages.scrollTop = messages.scrollHeight;
  }

  // ✅ Send on Enter key
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      sendToAdlatus();
    }
  });
</script>
`````
