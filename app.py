#streamlit run app.py
import os
import sys
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
import streamlit as st
import re
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# 引入你的后端逻辑
from src.task2_llm.graph import build_graph
from src.task2_llm.config import (
    LLM_MODEL_NAME, 
    DASHSCOPE_API_KEY, 
    DASHSCOPE_BASE_URL
)

# --- 页面配置 ---
st.set_page_config(
    page_title="Lord of the Rings: AI Storyteller",
    page_icon="🗡️",
    layout="centered"
)

# --- 辅助函数：选项生成 ---
def generate_interactive_options(story_text: str):
    """生成后续剧情选项"""
    llm = ChatOpenAI(
        model=LLM_MODEL_NAME,
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        temperature=0.7
    )
    
    prompt = f"""
    Based on the story below, generate 3 distinct next actions for the protagonist.
    Format strictly as:
    1. [Action text]
    2. [Action text]
    3. [Action text]
    
    Story context:
    {story_text[-2000:]} 
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content
        options = []
        lines = content.split('\n')
        for line in lines:
            match = re.match(r'^\d+\.\s*(.*)', line.strip())
            if match:
                options.append(match.group(1))
        return options if len(options) >= 3 else []
    except Exception:
        return []

# --- 初始化 Session State (记忆) ---
if "messages" not in st.session_state:
    st.session_state.messages = [] # 聊天记录
if "selected_book" not in st.session_state:
    st.session_state.selected_book = "The Fellowship of the Ring"
if "app" not in st.session_state:
    st.session_state.app = build_graph() # 初始化 LangGraph
if "last_draft" not in st.session_state:
    st.session_state.last_draft = "" # 用于生成上下文
if "current_options" not in st.session_state:
    st.session_state.current_options = [] # 存储当前的选项
if "adventure_ended" not in st.session_state:
    st.session_state.adventure_ended = False # 标记冒险是否结束

# --- 侧边栏：设置 (已移除 Clear History) ---
with st.sidebar:
    st.header("📚 Settings")
    book_choice = st.radio(
        "Choose Timeline:",
        ["The Fellowship of the Ring", "The Two Towers", "The Return of the King"]
    )
    # [新增] 文本长度控制
    st.markdown("---")
    st.subheader("✍️ Narrative Style")
    length_option = st.select_slider(
        "Response Length:",
        options=["Concise", "Balanced", "Epic"],
        value="Balanced"
    )
    
    # 将选项映射为具体的 Prompt 指令
    length_map = {
        "Concise": "Keep it short and punchy (approx 100-150 words). Focus on action.",
        "Balanced": "Standard novel pacing (approx 250-300 words). Balance dialogue and description.",
        "Epic": "Detailed and descriptive (approx 400+ words). Focus on atmosphere and internal monologue."
    }
    selected_instruction = length_map[length_option]
    # 如果切换书目，重置所有状态
    if book_choice != st.session_state.selected_book:
        st.session_state.selected_book = book_choice
        st.session_state.messages = []
        st.session_state.last_draft = ""
        st.session_state.current_options = []
        st.session_state.adventure_ended = False
        st.rerun()
    
    st.markdown("---")
    st.markdown("**Debug Info:**")
    st.caption(f"Current Book: {st.session_state.selected_book}")
    st.caption("Refresh page to reset fully.")

# --- 主界面 ---
st.title("🗡️ Middle-earth AI Storyteller")
st.markdown(f"*Currently adventuring in: **{st.session_state.selected_book}***")

# 1. 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 如果冒险已结束，显示结束语并停止渲染输入
if st.session_state.adventure_ended:
    st.success("The adventure has ended. Refresh the page to start a new journey!")
    st.stop()

# 2. 处理用户输入 (按钮 或 文本框)
user_input = None

# 如果有选项，显示选项按钮 + 退出按钮
if st.session_state.current_options:
    st.write("---")
    st.subheader("🎲 What do you do next?")
    
    # 渲染三个选项
    cols = st.columns(3)
    for i, option in enumerate(st.session_state.current_options):
        if cols[i].button(f"Option {i+1}", help=option, use_container_width=True):
            user_input = option
            
    # 显示完整的选项文本供参考
    for i, option in enumerate(st.session_state.current_options):
        st.caption(f"**{i+1}.** {option}")

    # [新增] 退出按钮
    st.write("") # 空一行
    if st.button("🏁 End Adventure / Quit", type="secondary", use_container_width=True):
        st.session_state.messages.append({"role": "assistant", "content": "**(The traveler decided to rest. The story ends here.)**"})
        st.session_state.adventure_ended = True
        st.rerun()

# 手动输入框 (允许用户自定义动作)
chat_input = st.chat_input("Describe an action or scene...")
if chat_input:
    user_input = chat_input

# 3. 核心逻辑处理
if user_input:
    # --- 用户回合 ---
    st.session_state.current_options = [] # 清空旧选项
    
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- AI 回合 ---
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # 构造 Context
        full_prompt = user_input
        if st.session_state.last_draft:
            short_memory = st.session_state.last_draft[-500:].replace("\n", " ")
            full_prompt = f"Previous context: ...{short_memory}\n\nCurrent Action: {user_input}"
        
        inputs = {
            "query": full_prompt,
            "book": st.session_state.selected_book,
            "style_guidelines": selected_instruction, # <--- 传入这个新参数
            "revision_count": 0,
            "is_final": False
        }

        # 思考过程可视化
        with st.status("🧙‍♂️ Agent is thinking...", expanded=True) as status:
            latest_draft = ""
            
            for output in st.session_state.app.stream(inputs):
                for key, value in output.items():
                    if key == "retriever":
                        st.write("🔍 **Retriever**: Consulting the Lore & Style Indexes...")
                    elif key == "writer":
                        st.write("✍️ **Writer**: Drafting story segment...")
                        if "draft" in value:
                            latest_draft = value["draft"]
                    elif key == "critic":
                        if value.get("is_final"):
                            st.write("✅ **Critic**: Draft approved!")
                            status.update(label="✨ Story Generation Complete!", state="complete", expanded=False)
                        else:
                            st.write("❌ **Critic**: Issues found. Requesting revision...")
                            st.caption(f"Feedback: {value.get('critique')[:100]}...")

        # 显示生成的故事
        if latest_draft:
            response_placeholder.markdown(latest_draft)
            st.session_state.messages.append({"role": "assistant", "content": latest_draft})
            st.session_state.last_draft = latest_draft
            
            # 生成新选项
            with st.spinner("🎲 Generating next options..."):
                new_options = generate_interactive_options(latest_draft)
                st.session_state.current_options = new_options
                st.rerun() 
        else:
            st.error("Something went wrong. Please try again.")