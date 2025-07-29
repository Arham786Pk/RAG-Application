import os
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

from loader import load_documents
from splitter import split_documents
from embedder import get_embedder

from langchain_community.vectorstores import FAISS
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# 🔐 Set your API key (or use st.secrets if deployed)
os.environ["MISTRAL_API_KEY"] = "pd5XlBkp4ocvzZKun7XhO3oHJjTT7xIl"  # Replace with your actual key

# Function: Build vectorstore
def build_vectorstore(file_path: str, db_path: str = "VectorStore"):
    documents = load_documents(file_path)
    chunks = split_documents(documents)
    embedder = get_embedder()
    db = FAISS.from_documents(chunks, embedder)
    db.save_local(db_path)
    return db

# Function: Ask a question
def ask_question(db_path: str, question: str):
    embedder = get_embedder()
    db = FAISS.load_local(db_path, embedder, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # 🟡 Get context
    relevant_docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    if not context.strip():
        return "I don't know."

    # Prompt
    llm = ChatMistralAI(model="mistral-small", temperature=0.3)
    prompt_template = PromptTemplate(
        template="""
        Answer the question based on the provided context.
        If the answer is not in the context, say "I don't know".

        Context:
        {context}

        Question:
        {question}
        """,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=False
    )

    result = chain.invoke({"query": question})
    return result["result"]

# ============ STREAMLIT UI =============

st.set_page_config(page_title="📚 RAG Application By HMA", layout="centered")
st.title("📄 Ask Questions from Your Document (RAG)")

uploaded_file = st.file_uploader("📂 Upload a document", type=["pdf", "docx", "txt", "csv", "html", "md", "json"])

if uploaded_file:
    with st.spinner("📦 Processing document..."):
        temp_path = os.path.join("temp_uploaded_file" + os.path.splitext(uploaded_file.name)[-1])
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        build_vectorstore(temp_path)

        st.success("✅ Vectorstore created!")

        # Enable question box
        question = st.text_input("❓ Ask a question from the document:")
        if question:
            with st.spinner("🤖 Thinking..."):
                answer = ask_question("VectorStore", question)
                st.markdown("### 💬 Answer:")
                st.success(answer)
else:
    st.info("Upload a document to get started.")
