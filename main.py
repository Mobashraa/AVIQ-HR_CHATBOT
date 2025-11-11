from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from qachain import upload_and_process, build_qa_chain
from langchain_utilss import qa_chain as base_chain

app = Flask(__name__)
CORS(app)

# Track current mode and chain
current_mode = "default"  
uploaded_chain = None
uploaded_index_path = None


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🤖 HR Policy Chatbot API is running successfully!"})


# Upload Document

@app.route("/upload", methods=["POST"])
def upload_file():
    global uploaded_index_path, uploaded_chain, current_mode
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    upload_path = os.path.join("uploaded_docs", file.filename)
    file.save(upload_path)

    # Process document and switch mode
    uploaded_index_path = upload_and_process(upload_path)
    uploaded_chain = build_qa_chain(uploaded_index_path)
    current_mode = "upload"

    return jsonify({"message": f"File '{file.filename}' processed successfully! Switched to Upload Mode."})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("question")
    if not query:
        return jsonify({"error": "No question provided"}), 400

    result = base_chain.invoke({"query": query})
    return jsonify({"answer": result["result"]})


#Switch Back to Default Mode

@app.route("/reset", methods=["POST"])
def reset_to_default():
    global uploaded_chain, current_mode
    uploaded_chain = None
    current_mode = "default"
    return jsonify({"message": " Switched back to Default Mode (HR Policy Data)."})


#  Ask Questions

@app.route("/ask", methods=["POST"])
def ask():
    global uploaded_chain, current_mode
    data = request.get_json()
    query = data.get("question", "").strip()

    if not query:
        return jsonify({"answer": "Please enter a valid question."})

    try:
        if current_mode == "upload" and uploaded_chain:
            result = uploaded_chain.invoke({"query": query})
        else:
            result = base_chain.invoke({"query": query})

        return jsonify({"answer": result["result"], "mode": current_mode})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    os.makedirs("uploaded_docs", exist_ok=True)
    os.makedirs("vectorstore", exist_ok=True)
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


