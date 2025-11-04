from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_utilss import qa_chain
import os
from flask_cors import CORS

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Initialize Flask app
app = Flask("HR_POLICY chatbot")
CORS(app)
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🤖 HR Policy Chatbot API is running successfully!"})

@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        # Parse request
        data = request.get_json()
        question = data.get("question", "")

        # API key verification (optional)
        x_api_key = request.headers.get("x-api-key")
        if API_KEY and x_api_key != API_KEY:
            return jsonify({"error": "Invalid API Key"}), 401

        # LangChain QA logic
        result = qa_chain.invoke({"query": question})
        answer = result["result"]

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

