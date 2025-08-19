import os
import json
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def load_retriever():
    """
    Load retriever from Chroma (index folder) + docstore.json
    """

    id_key = "doc_id"

    # Load vectorstore (Chroma)
    # Use synchronous client to avoid Streamlit async loop issues
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        async_client=False  # <--- important fix
    )

    vectorstore = Chroma(
        collection_name="multi_modal_rag",
        embedding_function=embedding_function,
        persist_directory="./index",  # must exist in repo
    )

    # Load docstore (parent mapping)
    store = InMemoryStore()
    docstore_path = os.path.join("index", "docstore.json")

    if os.path.exists(docstore_path):
        with open(docstore_path, "r", encoding="utf-8") as f:
            parent_docs = json.load(f)
        store.mset(list(parent_docs.items()))

    # Build retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    return retriever
