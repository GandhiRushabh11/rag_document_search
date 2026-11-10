from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model

load_dotenv()


class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    LLM_MODEL = "google_genai:gemini-3.5-flash"
    LLM_REQUEST_TIMEOUT = 20

    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    os.environ["USER_AGENT"] = "rag_document_search/1.0"

    # Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]

    @classmethod
    def get_llm(cls):
        os.environ["GOOGLE_API_KEY"] = cls.GOOGLE_API_KEY
        return init_chat_model(
            cls.LLM_MODEL,
            request_timeout=cls.LLM_REQUEST_TIMEOUT,
            retries=0,
        )
