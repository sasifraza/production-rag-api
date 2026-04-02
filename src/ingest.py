import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DOCS_PATH = "docs"
VECTORSTORE_PATH = "vectorstore"


def main():
    documents = []

    for file in os.listdir(DOCS_PATH):
        path = os.path.join(DOCS_PATH, file)

        if file.endswith(".txt"):
            loader = TextLoader(path)
            documents.extend(loader.load())

        elif file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            pdf_docs = loader.load()

            # keep only the main content pages, exclude references / author bios
            pdf_docs = [d for d in pdf_docs if d.metadata.get("page", 999) <= 3]

            documents.extend(pdf_docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTORSTORE_PATH)

    print("Vectorstore created")


if __name__ == "__main__":
    main()