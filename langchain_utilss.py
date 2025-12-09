
import os
from dotenv import load_dotenv
load_dotenv()


# from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader


# DATA_FOLDER = ["/content/Anti_Child_Labour_&Forced_Labour_Policy.pdf",
# "/content/Background_Verification_Policy.pdf",
# "/content/CodeOfConduct&Ethics_Policy.pdf",
# "/content/Data_Protection&_Privacy_Policy.pdf",
# "/content/Employment_Agreement_Policy.pdf",
# "/content/EqualOpportunity&Anti_Discrimination_Policy.pdf",
# "/content/Exit_Policy.pdf",
# "/content/Grievance_Redressal.pdf",
# "/content/IT_&Cybersecurity_Policy.pdf",
# "/content/Leave_Policy.pdf",
# "/content/Maternity_Leave_Policy.pdf",
# "/content/Performance_Improvement_Plan_policy.pdf",
# "/content/Posh_Policy.pdf",
# "/content/Whistleblower_Policy.pdf",
# "/content/Workplace_Safety&Health_Policy.pdf",
# "/content/old_leave_policy.pdf",
# "/content/New_Leave_Policy.pdf",
#                ]

data_folder = "Data"

# Load all PDFs from folder
pdf_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".pdf")]

docs = []
for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    docs.extend(loader.load())

print(f"Loaded {len(docs)} pages from {len(pdf_files)} PDFs.")

from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=250)
docs = splitter.split_documents(docs)
# print(f"Created {len(docs)} document chunks.")

# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)


vectorstore = FAISS.from_documents(docs, embeddings)

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# Initialize LLM
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    model="gpt-4o-mini", 
    temperature=0,
)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

#  Create the prompt template (you missed a parenthesis earlier)
prompt_template = """
You are an HR policy assistant for AventIQ.
Always respond in clear and professional English, regardless of the input language.
Use the following context to answer the user's question accurately and helpfully.
If the answer is not directly in the context, provide the best related information available —
never say "I don't know".Also give the beautified patterns of answers wherever possible.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate.from_template(prompt_template)

# Create QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},  
    return_source_documents=True,
)

print("🤖 HR Policy Chatbot ready!")
print("Type 'exit' to end the chat.")
print("------------------------------------")
if __name__ == "__main__":
 while True:
    query = input("\n👩‍💼 You: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Chatbot: Goodbye!")
        # memory.clear() 
        break

    result = qa_chain.invoke ({"query": query}) 
    print("🤖 Chatbot:", result["result"])



