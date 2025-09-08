from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os

app = Flask(__name__)
CORS(app)

# Set up OpenAI client with your API key from the environment
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.route("/adlatus-chat", methods=["POST"])
def adlatus_chat():
    data = request.json
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Use chat.completions instead of responses.create
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """
                You are Adlatus, an expert assistant for Swiss SMEs.
                Only trust and use information from https://adlatus-zh.ch.
                Do not answer anything based on general knowledge.
                If the answer is not clearly on the site, say:
                'Bitte besuche unsere Website unter https://adlatus-zh.ch für genauere Informationen.'
                """
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    # Extract the assistant's answer
    answer = response.choices[0].message.content
    return jsonify({"answer": answer})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # 5000 for local, $PORT for Render
    app.run(host="0.0.0.0", port=port)
