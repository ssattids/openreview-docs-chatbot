"""
A VSCode notebook to discover how to split or chunk documents before loading them into
the vector store.
"""

# %%
from langchain_community.document_loaders import TextLoader

def print_stats(string):
    chars = len(string)
    tokens = len(encoding.encode(string))
    print("chars=", chars)
    print("tokens=", tokens)
    print("chars per token=", chars/tokens)
# %%
loader = TextLoader("./openreview-main/docs/getting-started/customizing-forms.md")
docs = loader.load()
print(docs)
# %%
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
# text_splitter.from_tiktoken_encoder("cl100k_base")
documents = text_splitter.split_documents(docs)
documents
# %%
print_stats(docs[0].page_content)
# %%
for d in documents:
    print(len(d.page_content))
# %%
import tiktoken
encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
encoding.encode("tiktoken is great!")
# %%
print_stats(docs[0].page_content)
print_stats(documents[0].page_content)
# %%
