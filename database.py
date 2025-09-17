import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer
import chromadb

# --- Configuration ---
DATA_PATH = "Alzheimer_data"
MODEL_PATH = "model_cache" 
DB_PATH = "vector_db"
COLLECTION_NAME = "alzheimers_research"

# --- Loading Documents ---
print(f"Loading documents from '{DATA_PATH}'...")
documents = []
for filename in os.listdir(DATA_PATH):
    if filename.endswith(".txt"):
        filepath = os.path.join(DATA_PATH, filename)
        loader = TextLoader(filepath, encoding="utf-8")
        documents.extend(loader.load())
print(f"Loaded {len(documents)} documents.")

# --- Chunking Text ---
print("Splitting documents into chunks...")
# This is the missing line that defines the text_splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts_chunks = text_splitter.split_documents(documents)
list_of_texts = [chunk.page_content for chunk in texts_chunks]
print(f"Split into {len(list_of_texts)} chunks.")

# --- Load Model and Create Embeddings ---
print(f"Loading embedding model from path: {MODEL_PATH}...")
model = SentenceTransformer(MODEL_PATH)

print("Creating embeddings...")
embeddings = model.encode(list_of_texts, show_progress_bar=True)
print("Embeddings created successfully!")

# --- Store Embeddings in ChromaDB ---
print(f"Storing embeddings in ChromaDB collection '{COLLECTION_NAME}'...")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
ids = [f"doc_{i}" for i in range(len(list_of_texts))]

collection.add(
    documents=list_of_texts,
    embeddings=embeddings.tolist(),
    ids=ids
)
print("\nVector database created successfully!")