# RAG Embedding Demo
# © 2026 Ravindu Wijesundara
# https://github.com/ravindutw/rag-embedding-demo.git

import os
import embedding
from vector_db import MilvusDB
import handle_docs
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
LOCATION = os.environ.get("GCP_REGION")
MILVUS_COLLECTION_NAME = os.environ.get("MILVUS_COLLECTION_NAME")
MILVUS_HOST = os.environ.get("MILVUS_HOST")
MILVUS_UNAME = os.environ.get("MILVUS_UNAME")
MILVUS_PWD = os.environ.get("MILVUS_PWD")

embedding_model = embedding.init_embedding_model(PROJECT_ID, LOCATION)
vector_db = MilvusDB.init_vector_db(embedding_model, MILVUS_COLLECTION_NAME, MILVUS_HOST, MILVUS_UNAME, MILVUS_PWD)

def embedd():
    docs = handle_docs.load_docs()
    chunks = handle_docs.chunking(docs)
    embedding.add_to_vector_db(vector_db, chunks)

def search(query):
    result = vector_db.similarity_search(query, k=1)
    print(result)

#embedd()
search("Quantitative variables")

