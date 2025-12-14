import os
from typing import List, Dict, Tuple
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# 导入配置
from .config import (
    VECTOR_DB_DIR, 
    EMBEDDING_MODEL_NAME, 
    DASHSCOPE_API_KEY, 
    DASHSCOPE_BASE_URL
)

class LotrRetriever:
    def __init__(self):
        # 1. 初始化 Embedding 模型 (必须与存入时一致)
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            check_embedding_ctx_length=False
        )
        
        # 2. 连接现有的向量数据库
        if not os.path.exists(VECTOR_DB_DIR):
            raise FileNotFoundError(f"向量库不存在: {VECTOR_DB_DIR}，请先运行 data_ingestion.py")
            
        self.vector_store = Chroma(
            persist_directory=str(VECTOR_DB_DIR),
            embedding_function=self.embeddings
        )
        print(f"✅ Loaded Vector DB from {VECTOR_DB_DIR}")

    def retrieve_lore(self, query: str, book_name: str, k: int = 3) -> List[Document]:
        """
        检索剧情设定 (Lore)。
        强制过滤：type='lore' AND book=book_name
        """
        print(f"🔍 [LORE SEARCH] Query: '{query}' | Book: '{book_name}'")
        return self.vector_store.similarity_search(
            query,
            k=k,
            filter={
                "$and": [
                    {"type": {"$eq": "lore"}},
                    {"book": {"$eq": book_name}}
                ]
            }
        )

    def retrieve_style(self, query: str, k: int = 3) -> List[Document]:
        """
        检索文风参考 (Style)。
        强制过滤：type='style'
        注意：Style 通常不需要严格限制书名，但为了保持语境一致，也可以加上 book 限制。
        这里我们只限制 type，允许它参考全系列的文风。
        """
        print(f"🎨 [STYLE SEARCH] Query: '{query}'")
        return self.vector_store.similarity_search(
            query,
            k=k,
            filter={"type": "style"}
        )

    def get_combined_context(self, query: str, book_name: str) -> Dict[str, str]:
        """
        为 Writer Agent 准备组合上下文。
        返回格式化的字符串，方便直接塞入 Prompt。
        """
        # 1. 获取剧情事实
        lore_docs = self.retrieve_lore(query, book_name, k=3)
        lore_text = "\n\n".join([f"[Fact]: {d.page_content}" for d in lore_docs])
        
        # 2. 获取文风参考 (使用相同的 query，看原文是如何描述类似场景的)
        style_docs = self.retrieve_style(query, k=3)
        style_text = "\n\n".join([f"[Excerpt]: {d.page_content}" for d in style_docs])
        
        return {
            "lore_context": lore_text,
            "style_context": style_text,
            "raw_lore": lore_docs,
            "raw_style": style_docs
        }

# --- 独立验证模块 ---
if __name__ == "__main__":
    # 使用 python -m src.task2_llm.retriever 运行
    try:
        retriever = LotrRetriever()
        
        # 测试场景：Frodo 在风云顶 (Weathertop) 被戒灵刺伤
        # 这发生在第一部《魔戒现身》
        test_query = "Frodo gets stabbed by the Nazgul blade at Weathertop"
        test_book = "The Fellowship of the Ring"
        
        print("\n" + "="*50)
        print("🚀 Testing Dual-Track Retrieval")
        print("="*50)
        
        context = retriever.get_combined_context(test_query, test_book)
        
        print("\n📘 --- Retrieved LORE (Facts/Summaries) ---")
        print(context["lore_context"] if context["lore_context"] else "No lore found.")
        
        print("\n🖋️ --- Retrieved STYLE (Original Text) ---")
        print(context["style_context"] if context["style_context"] else "No style found.")
        
        # 测试防穿越功能：在第三部搜第一部的剧情，理论上应该找不到相关 Lore
        print("\n" + "="*50)
        print("🛡️ Testing Time-Travel Prevention")
        print("="*50)
        wrong_book = "The Return of the King"
        print(f"Attempting to search '{test_query}' in '{wrong_book}'...")
        
        wrong_ctx = retriever.get_combined_context(test_query, wrong_book)
        if not wrong_ctx["lore_context"]:
            print("✅ SUCCESS: Correctly prevented retrieving events from the wrong book!")
        else:
            print(f"❌ WARNING: Found metadata leak:\n{wrong_ctx['lore_context']}")
            
    except Exception as e:
        print(f"❌ Error during verification: {e}")