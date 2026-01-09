import os
import time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Resolve project root and data path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "data.csv"
DATA_FILE = os.environ.get("DATA_FILE", str(DEFAULT_DATA_PATH))

# Config
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/e5-base-v2")
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "data" / "chroma_db"))
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "ayurvedic-collection")

# Embeddings (HuggingFace) with normalization
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)


def load_dataframe(csv_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}.")
        raise SystemExit(1)


def build_documents_and_metadata(df: pd.DataFrame):
    documents = df['content (Sutra, Meaning, and Key Analysis)'].tolist()
    metadata_df = df[['sutra_name', 'primary_category', 'safety_level', 'target_dosha', 'advice_type']]
    metadatas = metadata_df.to_dict('records')

    # Maintain a copy of text in metadata for easy inspection
    for i, meta in enumerate(metadatas):
        meta['text_content'] = documents[i]

    return documents, metadatas


def index_to_chroma(documents, metadatas):
    vectordb = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )

    print(f"Adding {len(documents)} documents to Chroma collection '{CHROMA_COLLECTION_NAME}'...")
    vectordb.add_texts(texts=documents, metadatas=metadatas)
    # Chroma >=0.4.x persists automatically; no manual persist needed.
    print("Indexed documents into Chroma.")


def test_retrieval(query: str, k: int = 2):
    # Apply E5 query prefix if applicable
    effective_query = query
    if "e5" in EMBEDDING_MODEL.lower():
        effective_query = f"query: {query}"

    vectordb = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )

    print(f"\n--- Testing Retrieval ---\nQuery: {query}")
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    results = retriever.invoke(effective_query)
    for i, doc in enumerate(results, 1):
        print(f"\nTop {i} (score unknown - Chroma uses distance internally)")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Category: {doc.metadata.get('primary_category')} | Sutra: {doc.metadata.get('sutra_name')}")


if __name__ == "__main__":
    df = load_dataframe(DATA_FILE)
    docs, metas = build_documents_and_metadata(df)
    index_to_chroma(docs, metas)
    test_retrieval("What happens if I drink cold water immediately after eating food?")
