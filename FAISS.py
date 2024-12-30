from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# Initialize FAISS
def setup_faiss(data):
    embeddings = OpenAIEmbeddings()  # Replace with the model used for embedding generation
    vectorstore = FAISS.from_texts([item['description'] for item in data], embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore
