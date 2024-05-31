from langchain_community.vectorstores import FAISS

class VectorStore: 
    def __init__(self):
        self.vector_db = {}
        self.vector_length = 0

    def build(self, chunked_documents, embeddings):
        self.vector_db = FAISS.from_documents(chunked_documents, embeddings)
        return self.vector_db
    
    def get_vectors(self):
        return self.vector_db

    # TODO: Save to database
    def save_local(self, filename):
        self.vector_db.save_local(filename + "_db")

    # TODO: Load from database
    def load_local(self, filename, embeddings, allow_dangerous_deserialization=False):
        self.vector_db = FAISS.load_local(filename + "_db", embeddings, allow_dangerous_deserialization)
        return self.vector_db

    def similarity_search(self, query, k=5):
        return self.vector_db.similarity_search(query, k)
