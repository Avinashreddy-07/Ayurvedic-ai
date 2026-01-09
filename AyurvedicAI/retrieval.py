from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import settings

def get_vector_store() -> Chroma:
    """Initialize and return the Chroma Vector Store."""
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )
    return Chroma(
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )

def retrieve_from_qdrant(query: str, k: int = 5) -> List[Document]:
    """Retrieve documents from Qdrant."""
    # Apply E5 query prefix for better retrieval if using e5 models
    effective_query = query
    if "e5" in settings.EMBEDDING_MODEL.lower():
        effective_query = f"query: {query}"

    print(
        f"DEBUG: Querying Chroma collection '{settings.CHROMA_COLLECTION_NAME}' for: '{effective_query}'"
    )

    try:
        vector_store = get_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(effective_query)
        return docs
    except Exception as e:
        print(f"ERROR: Retrieval failed - {e}")
        return []