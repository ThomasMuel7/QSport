from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import SystemMessage
from tools import predict_nba, predict_foot, predict_tennis
import os

_agent = None

SYSTEM_MESSAGE = SystemMessage(content="""You are QSport, an expert sports prediction assistant.
ABSOLUTE RULES:
- When you use a tool, return ITS RESULT EXACTLY as-is, without modifying it, without interpreting it, without inventing numbers.
- NEVER change the percentages or team names returned by the tools.
- Do NOT add disclaimers like "predictions may vary".""")

def get_agent():
    global _agent
    if _agent is not None:
        return _agent

    llm = OllamaLLM(
        model=os.getenv("OLLAMA_MODEL", "mistral"),
        base_url=os.getenv("OLLAMA_HOST", "http://ollama:11434")
    )
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=10
    )
    tools = [predict_nba, predict_foot, predict_tennis]
    _agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        handle_parsing_errors=True,
        max_iterations=15,
        max_execution_time=300,
        agent_kwargs={"system_message": SYSTEM_MESSAGE}
    )
    return _agent
