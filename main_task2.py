#Frodo and Sam meet Galadriel for the first time
import os
import sys
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# 1. 加载 .env 环境变量 (必须在其他 LangChain 导入之前执行)
load_dotenv()
import sys
import re
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# 引入我们构建的图和配置
from src.task2_llm.graph import build_graph
from src.task2_llm.config import (
    LLM_MODEL_NAME, 
    DASHSCOPE_API_KEY, 
    DASHSCOPE_BASE_URL
)

BOOK_MAP = {
    "1": "The Fellowship of the Ring",
    "2": "The Two Towers",
    "3": "The Return of the King"
}

def generate_interactive_options(story_text: str):
    """
    生成选项，并返回一个 Python 列表，方便后续映射。
    """
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
    
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content
    
    # 使用正则提取选项文本
    options = []
    lines = content.split('\n')
    for line in lines:
        # 匹配 "1. xxxx" 或 "1.xxxx" 格式
        match = re.match(r'^\d+\.\s*(.*)', line.strip())
        if match:
            options.append(match.group(1))
            
    # 如果解析失败，返回原始内容作为单选项（容错）
    if len(options) < 3:
        return [content], content
        
    return options, content

def main():
    print("🤖 Initializing Middle-earth AI Agent...")
    app = build_graph()
    
    print("="*60)
    print("   🗡️  LORD OF THE RINGS: AI STORYTELLER (V2.0) 🛡️")
    print("="*60)
    
    while True:
        print("\n" + "="*40)
        print("📚 Select the Timeline (Book):")
        print("1. The Fellowship of the Ring")
        print("2. The Two Towers")
        print("3. The Return of the King")
        print("q. Quit Program")
        
        choice = input("\nYour Choice (1/2/3/q): ").strip().lower()
        if choice == 'q':
            sys.exit(0)
            
        selected_book = BOOK_MAP.get(choice)
        if not selected_book:
            continue
            
        print(f"\n📖 Context set to: {selected_book}")
        
        # --- 初始化记忆 ---
        # 我们用这个变量保存上一轮生成的文本，作为下一轮的背景
        previous_context = ""
        current_options = [] # 存储当前的选项列表
        
        user_query = input("\n🎬 Describe the starting scene:\n> ")
        
        # 进入冒险循环
        while True:
            if user_query.strip().lower() == 'menu':
                break

            # --- 处理数字输入 ---
            # 如果用户输入的是数字，尝试从选项列表中获取对应文本
            if user_query.isdigit() and current_options:
                idx = int(user_query) - 1
                if 0 <= idx < len(current_options):
                    selected_action = current_options[idx]
                    print(f"\n✅ You chose: {selected_action}")
                    # 将 Query 替换为具体的动作描述
                    user_query = selected_action
                else:
                    print("❌ Invalid number. Using input literally.")

            # ---  注入上下文 (Short-term Memory) ---
            # 为了防止 Agent 忘记刚才发生了什么，我们将上一段故事的最后部分拼接到 Query 中
            # 但这对检索器不友好，所以我们只把 Context 传给 Agent，或者构造一个复合 Query
            
            # 策略：构造一个包含上下文提示的 Query
            if previous_context:
                # 截取上一段故事的最后 500 字符作为“前情提要”
                short_memory = previous_context[-500:].replace("\n", " ")
                full_prompt = f"Previous context: ...{short_memory}\n\nCurrent Action: {user_query}"
            else:
                full_prompt = user_query

            print("\n⚙️  Agent is thinking...")
            print("-" * 60)
            
            inputs = {
                "query": full_prompt, # 使用带有记忆的 Prompt
                "book": selected_book,
                "revision_count": 0,
                "is_final": False
            }
            
            latest_draft = None 
            
            for output in app.stream(inputs):
                for key, value in output.items():
                    if key == "writer":
                        print(f"  👉 [Writer] Drafting...")
                    elif key == "critic":
                        if value.get("is_final"):
                            print("  👉 [Critic] ✅ Approved")
                        else:
                            print("  👉 [Critic] ❌ Rejected (Revising...)")
                    
                    if "draft" in value:
                        latest_draft = value["draft"]

            print("-" * 60)
            
            if latest_draft:
                print("\n✨ --- GENERATED STORY --- ✨\n")
                print(latest_draft)
                print("\n" + "="*30)
                
                # 更新记忆
                previous_context = latest_draft
                
                print("🎲 Suggested Actions:")
                # 解析选项列表
                opts_list, opts_text = generate_interactive_options(latest_draft)
                current_options = opts_list # 保存列表供下次映射
                
                # 打印带编号的选项
                for i, opt in enumerate(current_options):
                    print(f"{i+1}. {opt}")
                
                print("-" * 30)
                print("\nWhat do you do next?")
                print("💡 Type '1', '2', '3' OR type your own action.")
                print("   (Type 'menu' to go back, 'q' to quit)")
                
                user_query = input("\n> ")
                if user_query.strip().lower() == 'q':
                    sys.exit(0)
            else:
                print("❌ Error: No story generated.")
                break

if __name__ == "__main__":
    main()