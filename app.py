import os
import streamlit as st
from base64 import b64decode

# Set API key from Streamlit Cloud secrets if available
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retriever_loader import load_retriever

# -------- Load retriever (expects ./index/faiss_index + docstore.json) -------
retriever = load_retriever()

def parse_docs(docs):
    """Split base64-encoded images and texts"""
    b64, text = [], []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception:
            text.append(doc)
    return {"images": b64, "texts": text}

def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for t in docs_by_type["texts"]:
            context_text += str(t)

    prompt_template = f"""
    Answer the question based only on the following context,
    which can include text, tables, and image(s).

    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    for image in docs_by_type["images"]:
        prompt_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
        })

    return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])

rag_chain = (
    {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(build_prompt)
    | ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    | StrOutputParser()
)

# -------------------- UI --------------------
st.title("📄 Multi-Modal RAG with Gemini (FAISS)")
st.caption("Ask questions over your indexed text, tables, and images")

q = st.text_input("Your question")
if st.button("Submit") and q:
    with st.spinner("Thinking..."):
        answer = rag_chain.invoke(q)
    st.markdown("### 💬 Answer")
    st.write(answer)
