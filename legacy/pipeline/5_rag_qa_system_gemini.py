import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

if "GOOGLE_API_KEY" not in os.environ:
    print("🚨 Google API key not found in environment variables.")
    os.environ["GOOGLE_API_KEY"] = input("Please enter your Google API key: ")

TEXT_DIR = "text_output"
PERSIST_DIR = "db_gemini" # Use a new directory for the Gemini-based database

def create_or_load_database():
    """
    Creates or loads a vector database using Google's embedding model.
    """
    if os.path.exists(PERSIST_DIR):
        print("✅ Loading existing vector database...")
        # --- Use Google's embedding model ---
        vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
        return vectordb

    print("📄 Loading documents...")
    loader = DirectoryLoader(TEXT_DIR, glob="*.txt")
    documents = loader.load()

    if not documents:
        print("❌ No documents found.")
        return None

    print("✂️ Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    print("🧠 Creating new vector database with Google Embeddings...")
    # --- Use Google's embedding model ---
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=PERSIST_DIR)
    vectordb.persist()
    print("💾 Database saved to disk.")
    
    return vectordb

def create_qa_chain(vectordb):
    """
    Creates the RAG chain using the Gemini Pro model.
    """
    # --- Use Gemini Pro as the LLM ---
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

if __name__ == "__main__":
    db = create_or_load_database()
    
    if db:
        chain = create_qa_chain(db)
        print("\n✅ Gemini RAG Q&A System is ready. Ask a question or type 'exit' to quit.")
        
        # --- Interactive Loop ---
        while True:
            query = input("\n❓ Your Question: ")
            if query.lower() == 'exit':
                break
            
            print("...Thinking...")
            try:
                result = chain.invoke({"query": query})
                
                print("\n💡 Answer:")
                print(result["result"])
                
            except Exception as e:
                print(f"🔥 An error occurred: {e}")