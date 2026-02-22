import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# =========================================================
# 1. 核心開關與設定
# =========================================================
USE_RAG = True  # <--- 設為 True 啟用檢索，設為 False 則關閉
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BASE_MODEL = "Qwen/Qwen2-7B-Instruct"
VECTOR_DB_DIR = "faiss_index_hf"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# =========================================================
# 2. 載入向量資料庫 (僅在啟用 RAG 時載入，節省資源)
# =========================================================
db = None
if USE_RAG:
    print("模式：[RAG 已啟用] - 載入向量庫與 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    db = FAISS.load_local(VECTOR_DB_DIR, embeddings, allow_dangerous_deserialization=True)
else:
    print("模式：[純 LLM 模式] - 將不使用任何外部資料庫回答問題。")

# =========================================================
# 3. 載入 Qwen2 模型 (4-bit 量化)
# =========================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

print(f"正在載入 Qwen2 模型...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# =========================================================
# 4. 推理邏輯
# =========================================================
def get_answer(question):
    retrieved_docs = []
    
    # 根據開關決定 System Prompt
    if USE_RAG and db is not None:
        # 進行檢索
        docs = db.similarity_search(question, k=3)
        retrieved_docs = [doc.page_content for doc in docs] # 儲存內容以便後續列出
        
        context = "\n".join(retrieved_docs)
        system_prompt = f"你是一個專業助手。請根據以下提供事實資料回答問題，不要加入資料以外的資訊：\n\n{context}"
    else:
        system_prompt = "你是一個專業助手，請根據你的知識回答問題。"

    # 組合 Prompt
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    
    return answer, retrieved_docs # 同時回傳答案與檢索內容

# =========================================================
# 5. 執行對比測試
# =========================================================
if __name__ == "__main__":
    status = "ON" if USE_RAG else "OFF"
    print(f"\n系統就緒！目前 RAG 狀態：[{status}]")
    
    while True:
        query = input("\n" + "="*30 + "\n請輸入問題 (或輸入 exit 退出)：")
        if query.lower() in ['exit', 'quit']:
            break
            
        ans, sources = get_answer(query)
        
        # 顯示檢索到的資料 (Debug 用)
        if USE_RAG and sources:
            print("\n[檢索到的參考文本片段]:")
            for i, src in enumerate(sources):
                # 為了方便閱讀，只顯示前 150 字並加上邊框
                content_preview = src.replace('\n', ' ')[:150]
                print(f"  ({i+1}) {content_preview}...")
            print("-" * 30)

        print(f"\n[AI 最終回答]:\n{ans}")