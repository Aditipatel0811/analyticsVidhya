from sentence_transformers import SentenceTransformer

def generate_embeddings(data):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [item['description'] for item in data]
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings
