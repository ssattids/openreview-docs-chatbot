"""
Exploratory demo
"""
# %%
import langchain
# %%
# OPENAI_API_KEY = "INSERT OPENAI API KEY"
# %%
from langchain_openai import ChatOpenAI

llm = ChatOpenAI() # read from environment variable OPENAI_API_KEY
# llm = ChatOpenAI(api_key=OPENAI_API_KEY)

# %%
llm.invoke("how can langsmith help with testing?")


# %%
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])
# %%
chain = prompt | llm 
# %%
chain.invoke({"input": "how can langsmith help with testing?"})
# %%
from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()
# %%
chain = prompt | llm | output_parser
# %%
chain.invoke({"input": "how can langsmith help with testing?"})

# %%
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")

docs = loader.load()
# %%
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings() # read from environment variable OPENAI_API_KEY
# embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# %%
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
# %%
from langchain.chains.combine_documents import create_stuff_documents_chain

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)
# %%
from langchain_core.documents import Document

document_chain.invoke({
    "input": "how can langsmith help with testing?",
    "context": [Document(page_content="langsmith can let you visualize test results")]
})

# %%
from langchain.chains import create_retrieval_chain

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
# %%
