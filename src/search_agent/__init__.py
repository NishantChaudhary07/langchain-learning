from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

load_dotenv()

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
)

tools=[TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main() -> None:
    print("Hello from langchain-learning!")
    result = agent.invoke({"messages":[HumanMessage(content="What is the latest news about India?")]})
    print(f"Agent result: {result}")