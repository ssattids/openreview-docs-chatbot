from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentHandler:
    def __init__(self):
        self.documents = []
        self.chunked_documents = None

    def load(self, path):
        ## TODO: Find a better way to load the documents
        pathlist = Path(path).glob('**/*.md')
        
        for i, sub_path in enumerate(pathlist):
            # because path is object not string
            str_path = str(sub_path)
            print(str_path)

            loader = TextLoader(str_path)
            self.documents += loader.load()

            # if i == 1:
            #     break

    def chunk_documents(self, chunk_size=10000):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
        self.chunked_documents = text_splitter.split_documents(self.documents)
    
    def get_documents(self):
        return self.documents
    
    def get_chunked_documents(self):
        if self.chunked_documents is None:
            self.chunk_documents()
        return self.chunked_documents
