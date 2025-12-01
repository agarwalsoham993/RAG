import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

API_KEY = SYSTEM_API_KEY
os.environ["GEMINI_API_KEY"] = API_KEY

DOC_PATH = "all_about_docker.txt"
with open(DOC_PATH, "w") as f:
    f.write(DOC_CONTENT)

print(f"1. Knowledge Base created at: {DOC_PATH}")

loader = TextLoader(DOC_PATH)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
docs = text_splitter.split_documents(documents)

print(f"2. Documents loaded and split into {len(docs)} chunks.")

# We use GoogleGenerativeAIEmbeddings to convert into numerical vectors.
# FAISS (Facebook AI Similarity Search) for the in-memory vector store.

try:
    print("3. Creating embeddings and building FAISS vector store...")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    DB_FAISS_PATH = "faiss_index"           #saving the vector generated
    vectorstore.save_local(DB_FAISS_PATH)
    
    retriever = vectorstore.as_retriever()
except Exception as e:
    print(f"An error occurred during embedding/vector store creation: {e}")
    exit
