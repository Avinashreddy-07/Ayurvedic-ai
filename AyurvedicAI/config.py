import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Database
    DB_USERNAME: str = "postgres"
    DB_PASSWORD: str = os.environ['DB_PASSWORD']
    DB_NAME: str = os.environ['DB_NAME']
    DB_HOST: str = "localhost"
    DB_PORT: str = "5432"
    
    # AI Services (OpenRouter)
    OPENROUTER_API_KEY: str | None = os.environ.get('OPENROUTER_API_KEY')
    OPENROUTER_BASE_URL: str = os.environ.get('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
    # Models
    EMBEDDING_MODEL: str = os.environ.get('EMBEDDING_MODEL', 'intfloat/e5-base-v2')
    LLM_MODEL: str = os.environ.get('LLM_MODEL', 'openrouter/auto')

    # Vector DB (Chroma)
    CHROMA_PERSIST_DIR: str = os.environ.get('CHROMA_PERSIST_DIR', str(os.path.join(os.path.dirname(__file__), 'data', 'chroma_db')))
    CHROMA_COLLECTION_NAME: str = os.environ.get('CHROMA_COLLECTION_NAME', 'ayurvedic-collection')

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.DB_USERNAME}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?sslmode=disable"

settings = Settings()