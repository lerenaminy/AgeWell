import os
import pickle
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

def run_indexing():
    # 1. 初始化模型 (與 App 端的 BGE-M3 保持一致)
    # BGE-M3 支援 1024 維度，是目前 RAG 檢索中精確度極高的模型
    print("⏳ Loading BGE-M3 embedding model...")
    model = SentenceTransformer('BAAI/bge-m3')
    
    # 2. 批量讀取知識庫檔案 (包含 .md 與 .txt)
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

    # 3. 專業遞迴切片 (Recursive Character Splitting)
    # 我們設定較大的 chunk_size 以保留臨床論文的完整語境
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       
        chunk_overlap=150,    # 重疊區域確保切片點之間的邏輯不會中斷
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.create_documents(documents)
    chunk_texts = [c.page_content for c in chunks]
    print(f"🧩 Chunking complete: Generated {len(chunk_texts)} knowledge fragments.")

    # 4. 向量化 (Embedding)
    print("🧠 Generating embeddings... This may take a few minutes for 1100+ chunks...")
    # normalize_embeddings=True 確保向量在同一尺度，提升檢索穩定性
    embeddings = model.encode(chunk_texts, normalize_embeddings=True)

    # 5. 建立並儲存 FAISS 索引
    # 使用 IndexFlatL2 進行精確搜索
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    # 儲存索引檔案與文字映射表
    faiss.write_index(index, "health_index.faiss")
    with open("doc_map.pkl", "wb") as f:
        pickle.dump(chunk_texts, f)
    
    print("🚀 Success! Professional RAG Index (health_index.faiss) is ready.")

if __name__ == "__main__":
    run_indexing()