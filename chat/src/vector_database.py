from langchain_community.vectorstores import FAISS

class VectorDatabase: 
    def __init__(self):
        self.vectors = {}
        self.vector_length = 0

    def build(self, chunked_documents, embeddings):
        self.vectors = FAISS.from_documents(chunked_documents, embeddings)
        return self.vectors
    
    def get_vectors(self):
        return self.vectors

    # TODO: Save to database
    def save_local(self, filename):
        self.vectors.save_local(filename + "_db")

    # TODO: Load from database
    def load_local(self, filename, embeddings, allow_dangerous_deserialization=False):
        self.vectors = FAISS.load_local(filename + "_db", embeddings, allow_dangerous_deserialization)
        return self.vectors

    def similarity_search(self, query, k=5):
        return self.vectors.similarity_search(query, k)
