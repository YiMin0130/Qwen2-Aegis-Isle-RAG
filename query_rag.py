import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 1. 設定與 Ingestion 腳本相同的路徑與模型
VECTOR_DB_DIR = "faiss_index_hf"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def query_vector_db():
    # 2. 初始化相同的 Embedding 模型
    # 看到你的路徑有 5090，建議 device 改用 'cuda' 跑起來會飛快
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cuda'}, 
        encode_kwargs={'normalize_embeddings': True}
    )

    # 3. 載入本地向量資料庫 (允許危險解封裝是因為這是你自己產生的 pkl)
    if not os.path.exists(VECTOR_DB_DIR):
        print(f"錯誤：找不到資料庫目錄 {VECTOR_DB_DIR}，請先執行 ingestion 腳本。")
        return

    print(f"正在載入向量庫...")
    db = FAISS.load_local(VECTOR_DB_DIR, embeddings, allow_dangerous_deserialization=True)

    # 4. 進行互動式查詢
    print("\n--- 埃癸斯島資料查詢系統已就緒 (輸入 'exit' 退出) ---")
    while True:
        user_query = input("\n請輸入你的問題：")
        if user_query.lower() in ['exit', 'quit', '退出']:
            break
        
        # 檢索最相關的 3 個片段
        docs = db.similarity_search(user_query, k=3)
        
        print(f"\n[檢索到的參考內容]:")
        for i, doc in enumerate(docs):
            print(f"--- 來源片段 {i+1} ---")
            print(doc.page_content)
            print("-" * 20)

if __name__ == "__main__":
    query_vector_db()