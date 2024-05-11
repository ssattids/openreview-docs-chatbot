# %%
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pathlib import Path
from langchain_community.document_loaders import TextLoader

from langchain_openai import OpenAIEmbeddings
# %%
from langchain_openai import ChatOpenAI

"""
    Instantiate OPENAI Embeddings and ChatOpenAI classes
"""
embeddings = OpenAIEmbeddings() # currenty reading API KEY from environment variable
# declare llm to be used later
llm = ChatOpenAI()

# OPENAI_API_KEY = "INSERT YOUR OPENAPI KEY"
# embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
# llm = ChatOpenAI(api_key=OPENAI_API_KEY)

# %%
"""
    Read all the files in the openreview-main folder and load them as Document types
"""
documents = []
pathlist = Path("./openreview-main").glob('**/*.md')
for i, path in enumerate(pathlist):
    # because path is object not string
    str_path = str(path)
    print(str_path)

    loader = TextLoader(str_path)
    documents += loader.load()

    if i == 1:
        break

# %%
"""
    Chunk the documents
"""
# text_splitter = RecursiveCharacterTextSplitter()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
# text_splitter.from_tiktoken_encoder("cl100k_base")
chunked_documents = text_splitter.split_documents(documents)
print(f"Total chunked documents: {len(chunked_documents)}")
# %%
# declare the vectore store to be used
vector_db = FAISS.from_documents(chunked_documents, embeddings)
# %%
"""
    Save and load the vector store (OPTIONAL)
"""
vector_db.save_local("faiss_index")
vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
len(vector_db.index_to_docstore_id)
# %%
"""
    Query the vector store
"""
query = "I am having trouble claiming a profile. How can I claim a profile?"
queries_docs = vector_db.similarity_search(query)
print(len(queries_docs))
queries_docs
# %%
"""
    Use the queried documents as context and make a call to the chat model
"""
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)
answer = document_chain.invoke({
    "input": query,
    "context": queries_docs
})
print(answer)

# %%
