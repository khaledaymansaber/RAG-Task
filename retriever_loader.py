import os
import json
import asyncio
from langchain.vectorstores import FAISS
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def load_retriever():
    """
    Load retriever from FAISS (index folder) + docstore.json
    """
    id_key = "doc_id"

    # Ensure there is an event loop for gRPC async client
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Create embeddings
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load FAISS vectorstore
    index_path = "./index/faiss_index"
    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(index_path, embedding_function)
    else:
        vectorstore = FAISS.from_texts([], embedding_function)

    # Load docstore (simple dict)
    docstore = {}
    docstore_path = os.path.join("index", "docstore.json")
    if os.path.exists(docstore_path):
        with open(docstore_path, "r", encoding="utf-8") as f:
            docstore.update(json.load(f))

    # Build retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=id_key,
    )

    return retriever
