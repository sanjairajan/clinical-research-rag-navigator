import streamlit as st
import os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

# --- Configuration ---
DB_PATH = "vector_db"
MODEL_PATH = "model_cache"
LLM_MODEL = "phi3"
COLLECTION_NAME = "alzheimers_research"

# --- App UI ---
st.set_page_config(page_title="Medical Research Assistant 🩺", layout="wide")
st.title("🔬 Clinical Research Navigator")
with st.sidebar:
    st.header("🧠 About the App")
    st.markdown("A RAG chatbot using local models to answer questions about Alzheimer's research papers.")
    st.header("⚠️ Disclaimer")
    st.warning("This is for educational purposes only. Always consult a qualified healthcare professional.")

# --- Load Resources ---
@st.cache_resource
def load_resources():
    embedding_model = SentenceTransformer(MODEL_PATH)
    llm = Ollama(model=LLM_MODEL)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    return embedding_model, llm, collection

embedding_model, llm, collection = load_resources()

# --- RAG Prompt ---
prompt_template = PromptTemplate.from_template(
    """
    ### Instruction:
    You are an expert medical research assistant. Answer the user's question based ONLY on the provided context. If the context does not contain the answer, say so. Be concise.

    ### Context:
    {context}

    ### User Question:
    {question}

    ### Expert Answer:
    """
)

# --- Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main Logic ---
if query := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # 1. RETRIEVAL
            query_embedding = embedding_model.encode(query).tolist()
            results = collection.query(
                query_embeddings=[query_embedding], 
                n_results=5,
                include=['documents', 'metadatas'] # <-- CHANGE: Ask for metadata
            )
            retrieved_docs = results['documents'][0]
            retrieved_metadatas = results['metadatas'][0]
            
            # 2. AUGMENTATION
            context = "\n\n".join(retrieved_docs)
            full_prompt = prompt_template.format(context=context, question=query)

            # 3. GENERATION
            response = llm.invoke(full_prompt)
            st.markdown(response)
            
            # 4. DISPLAY SOURCES
            sources = set()
            for meta in retrieved_metadatas:
                if 'source' in meta:
                    sources.add(os.path.basename(meta['source']))
            
            if sources:
                with st.expander("Show Sources"):
                    for source in sorted(list(sources)):
                        st.write(f"- `{source}`")

    st.session_state.messages.append({"role": "assistant", "content": response})