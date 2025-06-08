import os
import gradio as gr
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.tools import tool
from langchain_sambanova import ChatSambaNovaCloud
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
You are a helpful AI Assistant designed to help users navigate the Indian Tax System.
You use clever interpretations of Indian tax laws to help reduce tax liabilities legally, advise smart business or product purchases, 
and guide users to increase financial growth while minimizing taxes.
Only respond to questions related to indian tax laws,Money,econmics.
Your role is Economist.You should behave like an Economist and try to uncover economist angle in every angle to leverage indian tasx system to pay less tax and make better business decisions
"""

def build_graph():
    llm = ChatSambaNovaCloud(
        model="Meta-Llama-3.3-70B-Instruct",
        temperature=0.7,
        top_p=0.01,
    )
    llm_with_tools = llm.bind_tools(tools)

    def assistant(state: MessagesState):
        if len(state["messages"]) >= 6:
            print("Reached recursion limit.")
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content="Return a final response based on all prior messages.")
            ] + state["messages"])
            return {"messages": state["messages"] + [response]}

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
    return builder.compile()

graph = build_graph()

def chat_fn(message, history):
    messages = []
    for user_msg, bot_msg in history:
        messages.append(HumanMessage(content=user_msg))
        messages.append(AIMessage(content=bot_msg))
    messages.append(HumanMessage(content=message))
    print(messages)

    output = graph.invoke({"messages": messages})
    final_reply = output["messages"][-1].content
    return final_reply

chat_interface = gr.ChatInterface(
    fn=chat_fn,
    title="Indian Tax Assistant",
    description="Ask your questions about Indian taxes, smart business choices, or minimizing taxes using legal tools.",
)

if __name__ == "__main__":
    chat_interface.launch()
