from langgraph.graph import StateGraph, END
from .agent_state import AgentState
from .nodes import retrieve_node, writer_node, critic_node

def build_graph():
    # 1. 创建图
    workflow = StateGraph(AgentState)
    
    # 2. 添加节点
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("critic", critic_node)
    
    # 3. 定义边 (Edges)
    workflow.set_entry_point("retriever")
    workflow.add_edge("retriever", "writer")
    workflow.add_edge("writer", "critic")
    
    # 4. 定义条件边 (Conditional Edges)
    def should_continue(state: AgentState):
        if state["is_final"]:
            return "end"
        elif state["revision_count"] > 2: # 最多重写 2 次，防止死循环
            print("⚠️ Max revisions reached. Accepting current draft.")
            return "end"
        else:
            return "rewrite"
            
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "end": END,
            "rewrite": "writer"
        }
    )
    
    return workflow.compile()