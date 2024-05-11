# %%
"""
An example of how to use the langchain library to load documents from a website.
"""
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader

from bs4 import BeautifulSoup as Soup

url = "https://docs.python.org/3.9/"
loader = RecursiveUrlLoader(
    url=url, max_depth=2, extractor=lambda x: Soup(x, "html.parser").text
)
docs = loader.load()
# %%
