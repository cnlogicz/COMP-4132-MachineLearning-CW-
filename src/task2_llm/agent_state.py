from typing import TypedDict, List, Annotated
import operator

class AgentState(TypedDict):
    # 用户输入
    query: str
    book: str
    style_guidelines: str
    # RAG 检索到的上下文
    lore_context: str
    style_context: str
    
    # 生成的内容
    draft: str
    critique: str
    
    # 状态标记
    revision_count: int  # 防止无限循环重写
    is_final: bool       # 是否通过质检