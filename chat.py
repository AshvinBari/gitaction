import streamlit as st
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

# ------------------------------
# Load models (free + local)
# ------------------------------
@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    return embeddings, llm

embeddings, llm = load_models()

st.title("🧠 CSV-based RAG Chatbot (LangChain + Free LLM)")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'Question' not in df.columns or 'Answer' not in df.columns:
        st.error("❌ CSV must contain 'Question' and 'Answer' columns.")
    else:
        st.success("✅ CSV loaded successfully!")

        # Combine question-answer into single text block
        df['text'] = df['Question'] + " — " + df['Answer']

        with st.spinner("⚙️ Creating embeddings and FAISS index..."):
            vectorstore = FAISS.from_texts(df['text'].tolist(), embedding=embeddings)

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        user_query = st.text_input("💬 Ask your question:")

        if st.button("Get Answer") and user_query:
            with st.spinner("🤖 Thinking..."):
                response = qa_chain.run(user_query)
            st.markdown(f"### 🧩 Answer:\n{response}")

else:
    st.info("Please upload a CSV file containing 'Question' and 'Answer' columns.")
