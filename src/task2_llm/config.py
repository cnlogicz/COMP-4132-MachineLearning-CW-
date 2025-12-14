import os
from pathlib import Path

# --- 路径配置 ---
# 获取项目根目录 (假设当前文件在 src/task2_llm/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = BASE_DIR / "chroma_db"

# --- 阿里百炼 (DashScope) 配置 ---
# 建议在终端执行: export DASHSCOPE_API_KEY="sk-318e......."
# 或者直接在这里填入: "sk-xxxxxxxxxxxx"
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
#DASHSCOPE_API_KEY = "sk-318e......" 

if not DASHSCOPE_API_KEY:
    raise ValueError("未找到 DASHSCOPE_API_KEY")

# 必须使用兼容模式的 Base URL
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# --- 模型参数 ---
# 生成模型: qwen-flash (便宜) 或 qwen-max (更强逻辑，适合 Writer)
LLM_MODEL_NAME = "qwen-flash-2025-07-28"
#LLM_MODEL_NAME = "qwen3-max"

# 向量模型: 阿里百炼提供的通用文本向量模型
# 注意: 不能使用默认的 openai 模型名，必须指定为阿里支持的模型
EMBEDDING_MODEL_NAME = "text-embedding-v2" 

# --- RAG 参数 ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50