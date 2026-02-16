from langchain_milvus import Milvus

class MilvusDB:

    @staticmethod
    def init_vector_db(embeddings, milvus_collection_name, milvus_host):

        vector_store = Milvus(
            embedding_function=embeddings,
            collection_name=milvus_collection_name,
            connection_args = {
                "uri": milvus_host
            }
        )

        return vector_store

