import os
import pickle
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

def run_indexing():
    print("⏳ Loading BGE-M3 embedding model...")
    model = SentenceTransformer('BAAI/bge-m3')
    
    documents = []
    kb_path = "knowledge_base/"
    
    if not os.path.exists(kb_path):
        print(f"❌ Error: Folder '{kb_path}' not found. Please create it first.")
        return

    files = [f for f in os.listdir(kb_path) if f.endswith((".md", ".txt"))]
    for file in files:
        with open(os.path.join(kb_path, file), 'r', encoding='utf-8') as f:
            documents.append(f.read())
    
    print(f"📖 Loaded {len(documents)} professional documents.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       
        chunk_overlap=150,   
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.create_documents(documents)
    chunk_texts = [c.page_content for c in chunks]
    print(f"🧩 Chunking complete: Generated {len(chunk_texts)} knowledge fragments.")

    print("🧠 Generating embeddings... This may take a few minutes for 1100+ chunks...")
    # normalize_embeddings=True
    embeddings = model.encode(chunk_texts, normalize_embeddings=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    faiss.write_index(index, "health_index.faiss")
    with open("doc_map.pkl", "wb") as f:
        pickle.dump(chunk_texts, f)
    
    print("🚀 Success! Professional RAG Index (health_index.faiss) is ready.")

if __name__ == "__main__":
    run_indexing()
