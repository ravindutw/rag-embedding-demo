import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

FOLDER_PATH = "documents"

def load_docs():
  if not os.path.exists(FOLDER_PATH):
    raise Exception("Directory not found")
  else:
    loader = DirectoryLoader(FOLDER_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

  print(f"Loaded {len(docs)} pages from PDF.")

  return docs


def chunking(docs):
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
  )
  all_splits = text_splitter.split_documents(docs)

  print(f"Split into {len(all_splits)} chunks.")

  return all_splits