import os

from dotenv import load_dotenv
load_dotenv()

class Settings:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")

    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_CLUSTER_URL = os.getenv("QDRANT_CLUSTER_URL")
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "enterprise_rag")

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")



settings = Settings()

