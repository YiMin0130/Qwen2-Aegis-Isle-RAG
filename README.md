# Aegis Isle RAG: 基於 Qwen2-7B 與 FAISS 的檢索增強生成實驗

這個專案是一個簡單而完整的 **RAG (Retrieval-Augmented Generation)** 實作系統。它結合了大型語言模型（Qwen2-7B）的推理能力與向量資料庫（FAISS）的精準檢索，旨在解決模型對虛構知識或特定私有資料產生的「幻覺」問題。

> [!IMPORTANT]
> **內容虛構申明**
> [cite_start]本專案中使用的所有檢索文章（`aegis_isle_corpus.txt`）及相關問答內容 [cite: 1, 2]，皆由 **Gemini** 虛構產生。其內容包含地理、生物、科技史等描述純屬虛傳，無任何現實參考價值。

## 🌟 核心特色
* [cite_start]**精準檢索**：使用 `sentence-transformers` 多國語言模型將 3,000 字的虛構語料轉為向量索引 [cite: 1]。
* **本地推理**：整合 **Qwen2-7B-Instruct** 模型，透過 4-bit 量化技術實現在本地端的高效推理。
* **RAG 開關對比**：內建 `USE_RAG` 變數，可一鍵切換模式，直觀對比純 LLM 與 RAG 增強後的回答品質差異。
* **引用透明化**：在 RAG 模式下，系統會自動列出檢索到的原始文本片段，方便驗證回答的真實性。

## 🛠️ 技術棧
* **LLM 模型**: Qwen2-7B-Instruct
* **Embedding 模型**: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
* **框架**: LangChain, Hugging Face Transformers
* **向量資料庫**: FAISS (Facebook AI Similarity Search)

## 📂 專案結構
```text
[cite_start]├── aegis_isle_corpus.txt      # 核心資料庫文章 (由 Gemini 虛構產生) [cite: 1]
[cite_start]├── QA.txt                     # 測試問題與標準答案集 [cite: 2]
├── vector_store_initializer.py # 文本切割與向量化腳本
├── rag_qwen_qa.py             # RAG 與純 LLM 對比問答主程式
├── query_rag.py               # 基礎檢索測試工具
└── faiss_index_hf/            # 生成的向量資料庫檔案 (執行後產生)
```

## 🚀 快速開始

### 1. 建立並啟用環境
建議使用 Conda 管理虛擬環境以確保依賴項純淨：

```bash
# 建立虛擬環境
conda create -n yimin python=3.10 -y

# 啟用虛擬環境
conda activate yimin
export PYTHONNOUSERSITE=1
```

### 2. 安裝依賴項
請依序執行以下安裝命令：

```bash
pip install -U sentencepiece protobuf safetensors
pip install -U torch transformers datasets peft accelerate bitsandbytes
pip install langchain langchain-community langchain-huggingface
pip install sentence-transformers faiss-cpu
```

### 3. 初始化向量資料庫
將虛構文章進行切割並存入 FAISS 索引：

```bash
python vector_store_initializer.py
```

### 4. 執行問答測試
執行主程式進入互動式對話介面。你可以在 rag_qwen_qa.py 中修改 USE_RAG = True/False 來切換模式：

```bash
python rag_qwen_qa.py
```

## 📊 效果演示
在開啟 RAG 模式時，系統不僅會給出答案，還會列出檢索來源，確保回答有所根據：

```bash
[檢索到的參考文本片段]:
  (1) 銀鬃貓的聽覺極為敏銳，其內耳構造中包含了一種特殊的骨質板塊，使它們能夠捕捉到頻率低至五赫茲的超低頻震動...
------------------------------
[AI 最終回答]:
根據提供的資料，銀鬃貓捕捉超低頻震動的生理構造是其內耳構造中的特殊骨質板塊。
```

## 📝 授權
本專案基於 MIT 授權。
