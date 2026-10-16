from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")



def create_embeddings(chunks):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(texts)

    return embeddings