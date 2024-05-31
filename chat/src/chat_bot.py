from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class ChatBot:
    def __init__(self, llm, vector_db, context_size=5):
        self.llm = llm
        self.vector_db = vector_db
        self.context_size = context_size

    def set_context_size(self, context_size):
        self.context_size = context_size

    def query(self, query):
        context = self.vector_db.similarity_search(query, self.context_size)

        prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

        document_chain = create_stuff_documents_chain(self.llm, prompt)
        return document_chain.invoke({
            "input": query,
            "context": context
        })

    