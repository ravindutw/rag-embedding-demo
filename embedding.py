from langchain_google_genai import GoogleGenerativeAIEmbeddings

def init_embedding_model(project_id, location):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        project=project_id,
        location=location,
        vertexai=True
    )

    return embeddings