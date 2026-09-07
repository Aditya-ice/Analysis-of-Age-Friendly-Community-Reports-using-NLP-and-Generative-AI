import os
from flask import Flask, render_template, request
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Configuration ---
# Ensure the Google API key is set
if "GOOGLE_API_KEY" not in os.environ:
    print("🚨 Google API key not found. Please set it as an environment variable.")
    exit()

PERSIST_DIR = "db_gemini"

# --- Load the RAG Chain (Do this once at startup) ---
print("🧠 Loading RAG chain and vector database...")
try:
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_function)

    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1, convert_system_message_to_human=True)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        return_source_documents=True
    )
    print("✅ RAG chain loaded successfully.")
except Exception as e:
    print(f"🔥 Failed to load RAG chain. Error: {e}")
    qa_chain = None

# --- Web Routes ---

@app.route('/')
def index():
    """Renders the main page with the search form."""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """Handles the user's question and returns the answer."""
    question = request.form.get('question')

    if not question:
        return render_template('index.html', error="Please ask a question.")

    if not qa_chain:
        return render_template('index.html', question=question, error="The RAG chain is not available. Please check server logs.")

    try:
        print(f"❓ Received Question: {question}")
        result = qa_chain.invoke({"query": question})
        answer = result.get("result", "Sorry, I couldn't find an answer.")
        print(f"💡 Sending Answer: {answer}")
        
        return render_template('index.html', question=question, answer=answer)

    except Exception as e:
        print(f"🔥 Error during question processing: {e}")
        return render_template('index.html', question=question, error="An error occurred while processing your question.")

if __name__ == '__main__':
    app.run(debug=True)