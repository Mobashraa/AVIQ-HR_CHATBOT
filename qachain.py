AZURE_OPENAI_API_KEY = "ELS7NnWuZ0AARkr0VgHwEVp1pS2qQpVt0QFrCVrjJ2tXpIhkwUicJQQJ99BHAC77bzfXJ3w3AAABACOGZotR"
AZURE_OPENAI_ENDPOINT = "https://msl-ai.openai.azure.com/"
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"

import os
# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {"pdf", "docx", "doc", "txt", "ppt", "pptx", "xls", "xlsx", "csv"}

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    UnstructuredCSVLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI


# -----------------------------
# 1. Load & Extract Documents
# -----------------------------
def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext in [".ppt", ".pptx"]:
        loader = UnstructuredPowerPointLoader(file_path)
    elif ext in [".xls", ".xlsx"]:
        loader = UnstructuredExcelLoader(file_path)
    elif ext == ".csv":
        loader = UnstructuredCSVLoader(file_path)
    else:
        raise ValueError(f"❌ Unsupported file type: {ext}")

    return loader.load()


# -----------------------------
#  2. Split & Embed Documents
# -----------------------------
def process_document(file_path, vectorstore_dir="data/vectorstore"):
    os.makedirs(vectorstore_dir, exist_ok=True)
    docs = load_document(file_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f" Split into {len(chunks)} chunks")

    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    index_path = os.path.join(vectorstore_dir, f"{base_name}_index")
    vectorstore.save_local(index_path)

    return index_path


# -----------------------------
#  3. Load FAISS Vectorstore
# -----------------------------
def load_vectorstore(index_path):
    if not os.path.exists(index_path):
        raise ValueError("Vectorstore not found. Please process documents first.")

    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)


# -----------------------------
#  4. Build QA Chain
# -----------------------------
def build_qa_chain(index_path):
    vectorstore = load_vectorstore(index_path)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt_template = """
You are an intelligent HR Policy Assistant for AventIQ. however if somebody uploads document you have to process it and give answers from it
give correct and precise answers based on the provided context .

If the answer is not found in the context, dont'say:
"I'm sorry, I couldn’t find the information related to that in the document."or " i dont know" instead give something meaningful related information.

Context:
{context}

Question:
{question}

Answer:
"""
    PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        model="gpt-4o-mini",
        temperature=0,
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return chain


# -----------------------------
#  5. Ask Questions
# -----------------------------
def ask_question(query, index_path):
    chain = build_qa_chain(index_path)
    result = chain.invoke({"query": query})
    return result["result"]
   
import os
import shutil

# Folders for uploaded documents and vectorstores
UPLOAD_DIR = "uploaded_docs"
VECTORSTORE_DIR = "vectorstore"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
def upload_and_process(file_path):
    base_name = os.path.basename(file_path)
    dest_path = os.path.join(UPLOAD_DIR, base_name)

    # ✅ Only copy if the file is not already inside uploaded_docs
    if os.path.abspath(file_path) != os.path.abspath(dest_path):
        shutil.copy(file_path, dest_path)
        print(f"📄 File copied to {dest_path}")
    else:
        print(f"⚠️ File already in upload directory, skipping copy.")

    # Process and store embeddings
    index_path = process_document(dest_path, vectorstore_dir=VECTORSTORE_DIR)
    print(f" Document processed and vectorstore saved at {index_path}")
    return index_path

# -----------------------------
#  6. Example Usage (Test)
# -----------------------------


if __name__ == "__main__":
    print("📂 Upload your document (local path):")
    file_path = input("Path: ")

    # Upload & process
    index_path = upload_and_process(file_path)

    # Build QA chain
    qa_chain = build_qa_chain(index_path)

    print("\n🤖 HR Policy Chatbot ready!")
    print("Type 'exit' to end the chat.")
    print("------------------------------------")

    while True:
        query = input("\n👩‍💼 You: ")
        if query.lower() in ["exit", "bye", "quit"]:
            print("👋 Chatbot: Goodbye!")
            break

        result = qa_chain.invoke({"query": query})
        print("🤖 Chatbot:", result["result"])
