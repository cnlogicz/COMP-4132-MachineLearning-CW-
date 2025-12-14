import re
import os
import shutil
from typing import List, Dict
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .config import (
    DATA_DIR, VECTOR_DB_DIR, 
    LLM_MODEL_NAME, EMBEDDING_MODEL_NAME, # 注意变量名变化
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL,
    CHUNK_SIZE, CHUNK_OVERLAP
)

class LotrDataPipeline:
    def __init__(self):
        # 1. 初始化 LLM (Qwen-Plus)
        self.llm = ChatOpenAI(
            model=LLM_MODEL_NAME,
            temperature=0.3,
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL
        )
        
        # 2. 初始化 Embedding (Text-Embedding-V2)
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            check_embedding_ctx_length=False
        )
        
        # 3. 初始化文本切分器 (这里是之前报错缺失的部分)
        # 用于 Style Index 的细粒度切分
        self.style_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
    def load_and_split_chapters(self, file_path: str, book_name: str) -> List[Dict]:
        """
        读取书籍并按章节进行粗切分 (Parent Documents)。
        注意：这里假设 txt 格式包含 'Chapter' 关键字，根据实际 txt 格式可能需要调整正则。
        """
        print(f"Loading {book_name}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f: # 备用编码
                text = f.read()

        # 简单的正则按章节切分 (假设格式为 "Chapter 1", "Chapter I" 等)
        # 这一步是为了获取 "上下文完整的章节" 用来生成 Lore Summary
        chapter_pattern = r'(Chapter\s+\d+|Chapter\s+[IVXLCDM]+)'
        parts = re.split(chapter_pattern, text)
        
        chapters = []
        current_chapter_title = "Prologue" # 默认前言
        
        # re.split 会保留分隔符，所以需要重新组合
        for part in parts:
            if re.match(chapter_pattern, part):
                current_chapter_title = part.strip()
            else:
                if len(part.strip()) < 100: continue # 跳过过短的片段
                chapters.append({
                    "text": part.strip(),
                    "metadata": {
                        "book": book_name,
                        "chapter": current_chapter_title
                    }
                })
        
        print(f"Found {len(chapters)} chapters in {book_name}")
        return chapters

    def generate_lore_summary(self, chapter_text: str) -> str:
        """
        使用 LLM 提取章节的事实摘要 (Lore Index)。
        """
        prompt = ChatPromptTemplate.from_template(
            """
            You are an expert editor creating a 'Lore Database' for Lord of the Rings.
            Summarize the following chapter text. 
            Focus ONLY on: plot progression, key events, character locations, and items obtained.
            Do NOT include flowery descriptions or dialogue unless it reveals a key fact.
            
            Text: {text}
            
            Summary:
            """
        )
        chain = prompt | self.llm | StrOutputParser()
        # 截断过长的章节文本以适应 Context Window (视模型而定)
        return chain.invoke({"text": chapter_text[:12000]})

    def process_all_books(self):
        """
        主流程：读取 -> 摘要(Lore) -> 切片(Style) -> 入库
        """
        books = [
            ("The Fellowship of the Ring", DATA_DIR / "01 - The Fellowship Of The Ring.txt"),
            ("The Two Towers", DATA_DIR / "02 - The Two Towers.txt"),
            ("The Return of the King", DATA_DIR / "03 - The Return Of The King.txt")
        ]
        
        documents_to_index = []

        for book_name, path in books:
            if not path.exists():
                print(f"Warning: File not found {path}")
                continue
                
            chapters = self.load_and_split_chapters(path, book_name)
            
            for i, chap in enumerate(chapters):
                print(f"Processing {book_name} - {chap['metadata']['chapter']}...")
                
                # --- Track 1: Lore Index (Summary) ---
                # 生成摘要
                summary_text = self.generate_lore_summary(chap['text'])
                lore_doc = Document(
                    page_content=summary_text,
                    metadata={
                        **chap['metadata'],
                        "type": "lore",  # 关键标签：用于检索事实
                        "source_type": "summary"
                    }
                )
                documents_to_index.append(lore_doc)
                
                # --- Track 2: Style Index (Original Chunks) ---
                # 对原文进行切片，用于模仿文风
                raw_docs = [Document(page_content=chap['text'], metadata=chap['metadata'])]
                style_chunks = self.style_splitter.split_documents(raw_docs)
                
                # 为 Style chunks 添加特定标签
                for chunk in style_chunks:
                    chunk.metadata.update({
                        "type": "style", # 关键标签：用于检索文风
                        "source_type": "original_text"
                    })
                documents_to_index.extend(style_chunks)

        return documents_to_index

    def build_vector_db(self, documents: List[Document]):
        """
        构建 Chroma 向量库（手动分批处理以适配阿里百炼 API 限制）
        """
        # --- 安全检查 ---
        if not documents:
            print("⚠️  Warning: No documents to index! Skipping Vector DB creation.")
            return

        # 清理旧数据库
        if os.path.exists(VECTOR_DB_DIR):
            print("Removing old vector database to rebuild...")
            shutil.rmtree(VECTOR_DB_DIR)
            
        print(f"Creating Vector DB with {len(documents)} documents...")
        
        # 1. 先初始化一个空的 Chroma 实例
        # 注意：这里不直接传入 documents，而是后面手动 add
        vector_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory=str(VECTOR_DB_DIR)
        )
        
        # 2. 手动分批写入 (阿里百炼限制 batch_size <= 25)
        batch_size = 20  # 设置为 20 以留有余地
        total_docs = len(documents)
        
        print(f"Start ingestion in batches of {batch_size}...")
        
        for i in range(0, total_docs, batch_size):
            # 切片获取当前批次
            batch = documents[i : i + batch_size]
            
            # 写入向量库
            vector_store.add_documents(batch)
            
            # 打印进度条效果 (使用 \r 回车不换行)
            print(f"Processed {min(i + batch_size, total_docs)}/{total_docs} documents...", end="\r")
            
        print("\n✅ Vector Database built successfully!")

if __name__ == "__main__":
    # 可以在这里测试运行
    pipeline = LotrDataPipeline()
    docs = pipeline.process_all_books()
    pipeline.build_vector_db(docs)