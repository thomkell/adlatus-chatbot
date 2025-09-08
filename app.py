from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.route("/adlatus-chat", methods=["POST"])
def adlatus_chat():
    data = request.json
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    response = client.responses.create(
        model="gpt-4o",
        instructions="""
            You are Adlatus, an expert assistant for Swiss SMEs.
            Only trust and use information from https://adlatus-zh.ch.
            Do not answer anything based on general knowledge.
            If the answer is not clearly on the site, say:
            'Bitte besuche unsere Website unter https://adlatus-zh.ch für genauere Informationen.'
        """,
        input=question,
        tools=["web_search"],
        store=False
    )

    for item in response.output:
        if item["type"] == "message":
            answer = item["content"][0]["text"]
            return jsonify({"answer": answer})

    return jsonify({"error": "No response from OpenAI"}), 500

if __name__ == "__main__":
    app.run(debug=True)
