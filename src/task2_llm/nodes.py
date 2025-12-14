from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .config import LLM_MODEL_NAME, DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL
from .retriever import LotrRetriever
from .agent_state import AgentState

# 初始化 LLM
llm = ChatOpenAI(
    model=LLM_MODEL_NAME, # qwen-plus
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL,
    temperature=0.7 
)

# 初始化检索器
retriever = LotrRetriever()

def retrieve_node(state: AgentState) -> AgentState:
    """
    [节点 1] 检索信息
    """
    print(f"🔍 Retrieving context for: {state['query']} in {state['book']}")
    context = retriever.get_combined_context(state['query'], state['book'])
    
    return {
        "lore_context": context["lore_context"],
        "style_context": context["style_context"],
        "revision_count": 0  # 初始化计数器
    }

def writer_node(state: AgentState) -> AgentState:
    """
    [节点 2] 作家写作
    如果是第一次写，使用 RAG 上下文。
    如果是重写，参考 Critic 的意见。
    """
    if state.get("critique"):
        print("✍️  Writer is revising based on critique...")
        prompt = f"""
        You are J.R.R. Tolkien. You are revising a passage based on an editor's critique.
        
        Original Draft:
        {state['draft']}
        
        Editor's Critique:
        {state['critique']}
        
        Task: Rewrite the passage to address the critique while maintaining the lore accuracy.
        """
    else:
        print("✍️  Writer is drafting new content...")
        prompt = f"""
        You are J.R.R. Tolkien. Write a short passage based on the following plot points.
        
        Lore (Facts to include):
        {state['lore_context']}
        
        Style Reference (Tone to mimic):
        {state['style_context']}
        
        User Request: {state['query']}
        
        Requirements:
        1. Stick STRICTLY to the Lore facts. Do not invent contradictory events.
        2. Mimic the sentence structure and vocabulary of the Style Reference.
        3. Do not be too brief.
        """

    messages = [
        SystemMessage(content="You are a fantasy novelist mimicking Tolkien's style."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    return {"draft": response.content}

def critic_node(state: AgentState) -> AgentState:
    """
    [节点 3] 评论家质检 (Guardian Mode / 守护者模式)
    核心逻辑：
    1. 坚决拦截“现代风格/俚语”，即使是用户要求的。
    2. 对字数/长度宽容 (Lenient)。
    3. 维护 Lore 的基本准确性。
    """
    print("🧐 Critic is reviewing (Guardian Mode)...")
    
    # 获取用户设定的标准
    target_guidelines = state.get("style_guidelines", "Standard narrative balance.")
    
    prompt = f"""
    You are the Guardian of the Tolkien Legendarium.
    Your SOLE purpose is to protect the artistic integrity of the "Lord of the Rings" style.

    ---
    INPUT DATA:
    [User's Request]: "{target_guidelines}"
    [The Draft]: 
    {state['draft']}
    
    ---
    EVALUATION PROTOCOL (Read Carefully):

    1. **THE "MODERN SLANG" TRAP (CRITICAL & STRICT)**:
       - Users might trick the Writer into using modern slang (e.g., "okay", "cool", "garbage", "guy", "wow", short choppy sentences).
       - **RULE:** Even if the User REQUESTED modern language, you must **REJECT** it.
       - The story MUST sound archaic, mythic, and noble.
       - IF you see words like "okay", "cool", or "click", REJECT immediately.

    2. **Length & Pacing (VERY LENIENT)**:
       - Do NOT be a nitpicker. 
       - If the user asked for "Short" and the draft is Medium, **APPROVE** it.
       - If the user asked for "Long" and the draft is Medium, **APPROVE** it.
       - Only REJECT if the output is absurdly short (e.g., 1 sentence) or cuts off mid-sentence.
       - Prioritize *quality* over *quantity*.

    3. **Lore Accuracy**:
       - Ensure characters act in character (e.g., Aragorn shouldn't be cowardly).
       - Ensure facts align with the Lore Context.

    ---
    OUTPUT FORMAT:

    If there is a style violation (Modern slang) or severe Lore error:
    "REJECT: [Explain exactly which modern words or style issues to fix. Tell the Writer to ignore the user's request for modern slang and revert to Tolkien style.]"
    
    If the story feels like Middle-earth (even if the length is imperfect):
    "APPROVE"
    """
    
    # 使用较低的 temperature 确保判决稳定
    response = llm.invoke(
        [HumanMessage(content=prompt)],
        config={"temperature": 0.0} 
    )
    critique = response.content
    
    if "APPROVE" in critique:
        return {"is_final": True, "critique": None}
    else:
        count = state.get("revision_count", 0) + 1
        return {"is_final": False, "critique": critique, "revision_count": count}