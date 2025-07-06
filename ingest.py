import os
from dotenv import load_dotenv
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker
from langchain_docling.loader import ExportType
from qdrant_client import QdrantClient

# Load environment
load_dotenv()
qdrant_url = os.getenv("QDRANT_URL_LOCALHOST")
print(f"Qdrant URL: {qdrant_url}")

# PDF input file
FILE_PATH = "./data/test2.pdf"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "rag"

def create_vector_database():
    # Delete old collection if it exists
    try:
        client = QdrantClient(url=qdrant_url)
        existing_collections = client.get_collections().collections
        if any(col.name == COLLECTION_NAME for col in existing_collections):
            client.delete_collection(collection_name=COLLECTION_NAME)
            print(f" Deleted existing '{COLLECTION_NAME}' collection.")
    except Exception as e:
        print(f" Failed to delete collection: {e}")

    # Load the PDF with Docling
    loader = DoclingLoader(
        file_path=FILE_PATH,
        export_type=ExportType.DOC_CHUNKS,  
        chunker=HybridChunker(tokenizer=EMBED_MODEL_ID),
    )
    docling_documents = loader.load()

    # Optional: write parsed content to markdown
    with open("data/output_docling.md", "w", encoding="utf-8") as f:
        for doc in docling_documents:
            f.write(doc.page_content + '\n\n')

    # Embed & upload to Qdrant
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)
    vectorstore = QdrantVectorStore.from_documents(
        documents=docling_documents,
        embedding=embedding,
        url=qdrant_url,
        collection_name=COLLECTION_NAME,
    )

    print("Vector DB created successfully with full content (including tables) from test2.pdf!")

if __name__ == "__main__":
    create_vector_database()
