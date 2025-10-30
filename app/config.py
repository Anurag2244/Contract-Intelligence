import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    # PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
    UPLOAD_DIR = "./data/documents"

settings = Settings()