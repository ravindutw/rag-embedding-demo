import os
import embedding
from vector_db import MilvusDB
from pymilvus import utility

PROJECT_ID = "sliit-labs-and-projects"
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "global")

key_path = "keys/sliit-labs-and-projects-0696ba00c169.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

embedding_model = embedding.init_embedding_model(PROJECT_ID, LOCATION)

vector_db = MilvusDB.init_vector_db(embedding_model)

result = vector_db.similarity_search("test", k=1)

print(result)