from flask import Flask, request, jsonify
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from src.chat_bot import ChatBot
from src.document_handler import DocumentHandler
from src.vector_database import VectorDatabase

OPENAI_API_KEY = ""
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)
document_handler = DocumentHandler()
document_handler.load("./documents")
chunked_documents = document_handler.get_chunked_documents()
vector_database = VectorDatabase()
vectors = vector_database.build(chunked_documents, embeddings)
chat_bot = ChatBot(llm, vectors)

@app.route("/query", methods=["POST"])
def query():
    if request.is_json:
        data = request.json

        answer = chat_bot.query(data["query"])
        return jsonify({"answer": answer}), 200
    else:
        return jsonify({"error": "Request must contain JSON data"}), 400
    

if __name__ == "__main__":
    app.run(debug=True)
