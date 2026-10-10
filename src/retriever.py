from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from .ingest import load_documents, split_documents


VECTORSTORE_PATH = "vectorstore/faiss_index"


def build_vectorstore():
    # 1. Load PDF documents
    documents = load_documents()

    # 2. Split documents into chunks
    chunks = split_documents(documents)

    print(f"Total chunks: {len(chunks)}")

    # 3. Create embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Convert chunks into vectors and create FAISS index
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # 5. Save FAISS index locally
    vectorstore.save_local(VECTORSTORE_PATH)

    print("\nFAISS vector store created successfully.")
    print(f"Saved at: {VECTORSTORE_PATH}")

    return vectorstore

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def test_retrieval(vectorstore):
    query = query = "What was NVIDIA's net income in fiscal year 2024?"
    results = vectorstore.similarity_search_with_score(
        query,
        k=10
    )

    print("\n==============================")
    print("Retrieval Test")
    print("==============================")

    print(f"\nQuestion: {query}")

    for i, (document, score) in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print("Score:", score)
        print("Source:", document.metadata.get("source"))
        print("Page:", document.metadata.get("page"))
        print("Content:")
        print(document.page_content[:500])




if __name__ == "__main__":

    vectorstore = load_vectorstore()

    test_retrieval(vectorstore)