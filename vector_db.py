from langchain_milvus import Milvus

class MilvusDB:

    @staticmethod
    def init_vector_db(embeddings):
        vector_store = Milvus(
            embedding_function=embeddings,
            collection_name="vector_db",
            connection_args = {
                "uri": "http://dev-server.gcp.ravinduw.com:19530"
            }
        )

        return vector_store



"""
connection_args={
                "host": "dev-server.gcp.ravinduw.com",
                "port": "19530"
            }
"""