from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


DOCUMENTS_DIR = Path("data/documents")


def load_documents():
    documents = []

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files.")

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()

        documents.extend(docs)

        print(f"  Pages loaded: {len(docs)}")

    return documents


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":

    # Step 1: Load PDF pages
    documents = load_documents()

    # Step 2: Split pages into chunks
    chunks = split_documents(documents)

    print("\n==============================")
    print("Document chunking completed")
    print("==============================")

    print(f"Total pages: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")

    if chunks:

        print("\nFirst chunk metadata:")
        print(chunks[0].metadata)

        print("\nFirst chunk:")
        print(chunks[0].page_content[:1000])