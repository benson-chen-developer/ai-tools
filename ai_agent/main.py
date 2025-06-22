from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@tool
def friends(name: str) -> str:
    """Useful for when asking questions about my friend. The input should be the name of the friend."""
    
    data = {
        "Shawn": "Shawn works as a car mechanic. He is really good with sedans and Toyotas.",
        "Steven": "Steven works as a waiter. He js really good at cooking seafood and pasta.",
        "Mike": "Mike is a musician. He enjoys artists like Michael Jackson."
    }

    return data.get(name, f"I don't have any information about {name}.")

def main():
    model = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0)

    tools = [friends]
    agent_executor = create_react_agent(model, tools)

    print("Hello! I'm your AI assistant. Type 'quit' to exit.")
    print("You can ask me to perform calculations or chat with me.")

    while True:
        user_input = input("You: ").strip()

        if user_input == "quit":
            break

        print("Assistant: ", end="")
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]}
        ):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    print(message.content, end="")
        print()

if __name__ == "__main__":
    main()