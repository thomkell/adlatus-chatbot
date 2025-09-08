from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os

app = Flask(__name__)
CORS(app)

# Hole API-Key aus Umgebungsvariable
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.route("/adlatus-chat", methods=["POST"])
def adlatus_chat():
    data = request.json
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Du bist Adlatus, ein digitaler Assistent für Schweizer KMUs. "
                        "Antworte nur mit Informationen von https://adlatus-zh.ch. "
                        "Wenn du dir nicht sicher bist, sag: "
                        "'Bitte besuche unsere Website unter https://adlatus-zh.ch für genauere Informationen.'"
                    )
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        answer = response.choices[0].message.content
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # default for local
    app.run(host="0.0.0.0", port=port)
