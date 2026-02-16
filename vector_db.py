from langchain_milvus import Milvus

def init_vector_db(embeddings):
    vector_store = Milvus(
        embedding_function=embeddings,
        collection_name="vector_db",
        connection_args={
            "host": "18.223.10.55",
            "port": "19530"
        }
    )

    return vector_store