import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ─── CONFIG ────────────────────────────────────────────────
KB_PATH    = "data/knowledge_base/"
FAISS_PATH = "data/processed/faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def build_vector_store():
    print("📚 Loading knowledge base documents...")

    # Load all .txt files from knowledge base folder
    loader = DirectoryLoader(
        KB_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    print(f"   Loaded {len(documents)} documents")

    # Split into chunks
    print("✂️  Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = 500,
        chunk_overlap = 50,
        separators    = ["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"   Created {len(chunks)} chunks")

    # Show sample chunk
    print(f"\n📄 Sample chunk:\n{chunks[0].page_content[:200]}\n")

    # Load embedding model
    print(f"🔢 Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name      = EMBED_MODEL,
        model_kwargs    = {"device": "cpu"},
        encode_kwargs   = {"normalize_embeddings": True}
    )
    print("   ✅ Embedding model loaded")

    # Build FAISS index
    print("🗂️  Building FAISS vector store...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Save to disk
    os.makedirs("data/processed", exist_ok=True)
    vector_store.save_local(FAISS_PATH)
    print(f"   ✅ FAISS index saved to {FAISS_PATH}")
    print(f"   Total vectors indexed: {vector_store.index.ntotal}")

    return vector_store


def load_vector_store():
    """Load existing FAISS index from disk."""
    embeddings = HuggingFaceEmbeddings(
        model_name    = EMBED_MODEL,
        model_kwargs  = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True}
    )
    vector_store = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"✅ FAISS index loaded — {vector_store.index.ntotal} vectors")
    return vector_store


if __name__ == "__main__":
    vs = build_vector_store()

    # Test similarity search
    print("\n🔍 Testing similarity search...")
    query   = "What should I do if there is a port strike?"
    results = vs.similarity_search(query, k=3)

    print(f"\nQuery: '{query}'")
    print("─" * 50)
    for i, doc in enumerate(results):
        print(f"\n[Result {i+1}]")
        print(doc.page_content[:300])