from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from tools import predict_nba, predict_foot, predict_tennis
import os

def get_agent():
    llm = OllamaLLM(
        model=os.getenv("OLLAMA_MODEL", "mistral"),
        base_url=os.getenv("OLLAMA_HOST", "http://ollama:11434")
    )
    tools = [predict_nba, predict_foot, predict_tennis]
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True, max_iterations=15, max_execution_time=300
    )
