from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

DB_FAISS_PATH = "faiss_index"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore = FAISS.load_local(
    folder_path=DB_FAISS_PATH,
    embeddings=embeddings,
    allow_dangerous_deserialization=True 
)

retriever = vectorstore.as_retriever()

# gemini model 
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-09-2025",
    temperature=0.1
)

#creating the retrieval model
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

#Testing:
# expected answer = table 1 compare features of different containerized and virtual machine technologies. 
query = "explain table 1?"

try:
    result = qa_chain.invoke({"query": query})
    print("\n--- RAG Result ---")
    print(f"Answer: {result['result']}")
    
    #citation
    print("\n cited_via")
    for i, doc in enumerate(result['source_documents']):
        print(f"Source {i+1} (Source File: {doc.metadata.get('source', 'N/A')}):")
        print(f"  Content snippet: \"{doc.page_content.strip()}\"")
        print("-" * 20)

except Exception as e:
    print(f"\nAn error occurred {e}")

