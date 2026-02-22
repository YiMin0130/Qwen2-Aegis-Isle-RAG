import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 設定檔案路徑
INPUT_FILE = "aegis_isle_corpus.txt"
VECTOR_DB_DIR = "faiss_index_hf"

def load_text_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到檔案：{file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def create_vector_db(file_path, db_path):
    # 1. 讀取外部文本
    print(f"正在讀取檔案：{file_path}...")
    content = load_text_file(file_path)

    # 2. 設定文本切割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    
    chunks = text_splitter.split_text(content)
    docs = [Document(page_content=chunk) for chunk in chunks]
    print(f"文本已切割完成，總共生成 {len(docs)} 個 Chunks。")

    # 3. 使用 Hugging Face 模型初始化 Embedding
    # 模型推薦：
    # - "shibing624/text2vec-base-chinese" (針對中文優化)
    # - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (多語言支援)
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    print(f"正在載入 Hugging Face 模型: {model_name}...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        # 如果你有 NVIDIA 顯卡，可以改為 "cuda"，否則用 "cpu"
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 4. 建立與儲存向量資料庫
    print("正在轉換向量 (使用本地 CPU 運算)...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(db_path)
    print(f"向量庫已成功儲存至資料夾: {db_path}")

if __name__ == "__main__":
    try:
        create_vector_db(INPUT_FILE, VECTOR_DB_DIR)
    except Exception as e:
        print(f"執行出錯：{e}")