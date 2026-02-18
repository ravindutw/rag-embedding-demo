from langchain_milvus import Milvus

class MilvusDB:

    @staticmethod
    def init_vector_db(embeddings, milvus_collection_name, milvus_host, milvus_un, milvus_pwd):

        vector_store = Milvus(
            embedding_function=embeddings,
            collection_name=milvus_collection_name,
            connection_args = {
                "uri": milvus_host,
                "user": milvus_un,
                "password": milvus_pwd,
                "secure": True
            }
        )

        return vector_store

