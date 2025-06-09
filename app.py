import os
import gradio as gr
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, MessagesState,END
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.tools import tool
from langchain_sambanova import ChatSambaNovaCloud
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


load_dotenv()
sambanova_api_key = os.getenv("NOVA_API_KEY")
os.environ["SAMBANOVA_API_KEY"] = sambanova_api_key

@tool
def web_search(query: str) -> str:
    """Use this tool to perform real-time web search."""
    search_docs = TavilySearchResults(max_results=3).invoke(input=query)
    return {"web_results": search_docs}

tools = [web_search]

system_prompt = """
You are a Financial AI Assistant specialized in Indian Tax Systems.
Your goal is to help users make legal, optimized decisions for reducing taxes, 
growing businesses, and purchasing smartly. Use Indian tax law smartly and ethically.
Use web_search tool only if you need.otherwise don;t use tools.
Only respond to questions related to indian tax laws,Money,econmics.
Your role is Economist.You should behave like an Economist and try to uncover economist angle in every angle to leverage indian tasx system to pay less tax and make better business decisions
"""

# Build LangGraph with Memory Support
def build_graph():
    llm = ChatSambaNovaCloud(
        model="Meta-Llama-3.3-70B-Instruct",
        temperature=0.7,
        top_p=0.01,
    )
    llm_with_tools = llm.bind_tools(tools)

    def assistant(state: MessagesState):
        print(state)
        response = llm_with_tools.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content="Return a final response based on all prior messages.")
        ] + state["messages"])
        return {"messages": state["messages"] + [response]}
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    memory = MemorySaver()

    return builder.compile(checkpointer=memory)

graph = build_graph()
import uuid
def chat_fn(user_message, history):
    messages = []
    for entry in history:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            user, bot = entry
            messages.append(HumanMessage(content=user))
            messages.append(AIMessage(content=bot))
    messages.append(HumanMessage(content=user_message))

    try:
        output = graph.invoke({"messages": messages}, config={"configurable": {"thread_id": uuid.uuid4}})
        reply = output["messages"][-1].content
    except Exception as e:
        reply = f"❌ Error: {str(e)}"
    return reply

with gr.Blocks(theme=gr.themes.Soft(), title="Indian AI Tax Assistant") as demo:
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("""
                <div style="display: flex; align-items: center; gap: 12px;">
                    <h1 style="font-size: 28px;">Indian AI Tax Assistant </h1>
                </div>
               
            """)
            gr.Markdown("> ✅ <span style='color:green;'>LLM: Meta-Llama-3.3-70B (SambaNova)</span> | 🛠 Tools Enabled: Web Search")

            chat = gr.ChatInterface(
                fn=chat_fn,
                type = "messages",
                chatbot=gr.Chatbot(
                    label="Tax Advisor",
                    type = "messages",
                ),
                textbox=gr.Textbox(
                    label="Ask a Tax Question"
                ),
                title="💬 Chat with Indian Tax AI",
               
            )

           
    gr.Markdown("---")

if __name__ == "__main__":
    demo.launch()