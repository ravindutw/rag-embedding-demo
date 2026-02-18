import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

FOLDER_PATH = os.environ.get("FOLDER_PATH")

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
    chunk_size=500,  # chunk size (characters)
    chunk_overlap=50,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
  )
  all_splits = text_splitter.split_documents(docs)

  print(f"Split into {len(all_splits)} chunks.")

  return all_splits