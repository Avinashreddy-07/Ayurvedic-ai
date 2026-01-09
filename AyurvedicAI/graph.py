from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI


from config import settings
from prompts import SYSTEM_PROMPT
from tools import ALL_TOOLS

# 1. Define State
class BotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 2. Initialize Model (OpenRouter via OpenAI-compatible client)
# Validate API key presence to provide a clear error instead of KeyError
if not settings.OPENROUTER_API_KEY:
    raise RuntimeError(
        "Missing OPENROUTER_API_KEY. Set it in your .env (and restart) "
        "or export it in the environment."
    )

llm = ChatOpenAI(
    model=settings.LLM_MODEL,
    api_key=settings.OPENROUTER_API_KEY,
    base_url=settings.OPENROUTER_BASE_URL,
)
llm_with_tools = llm.bind_tools(tools=ALL_TOOLS)

# 3. Define Nodes
def get_response(state: BotState) -> dict:
    """
    Processes the state and generates a response using the LLM.
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state['messages']
    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}

# 4. Build Graph
def build_graph():
    graph_builder = StateGraph(BotState)
    tool_node = ToolNode(ALL_TOOLS)

    graph_builder.add_node('get_response', get_response)
    graph_builder.add_node('tools', tool_node)

    graph_builder.add_edge(START, 'get_response')
    graph_builder.add_conditional_edges('get_response', tools_condition)
    graph_builder.add_edge('tools', 'get_response')

    return graph_builder
