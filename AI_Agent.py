from typing import Any, List
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain.agents import create_agent

from dotenv import load_dotenv

load_dotenv()

@tool
def write_json(filepath: str, data: Any) -> str:
    """Write a Python object (list/dict/etc.) as JSON to a file with pretty formatting."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return f"Successfully wrote JSON data to '{filepath}' ({len(json.dumps(data))} characters)."
    except Exception as e:
        return f"Error writing JSON: {str(e)}"

@tool
def read_json(filepath: str) -> str:
    """Read and return the contents of a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    except FileNotFoundError:
        return f"Error: File '{filepath}' not found."
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in file - {str(e)}"
    except Exception as e:
        return f"Error reading JSON: {str(e)}"

TOOLS = [write_json, read_json]

llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

SYSTEM_PROMPT = (
    "You are DupeRemove, a helpful assistant that removes duplicated information from data sets in JSONs."
    "Don't answer any questions that are not related to your specialty."
    "To find duplicates, you need: the data file (json) and whether the user wants to edit the file, save it to a new one, or just point out the duplicates in the console."
    "Ask the user questions about the required info if they haven't already provided them."
)

agent = create_agent(llm, TOOLS, system_prompt=SYSTEM_PROMPT)

def run_agent(user_input: str, history: List[BaseMessage]) -> AIMessage:
    """Single-turn agent runner with automatic tool execution via LangGraph."""
    try:
        result = agent.invoke(
            {"messages": history + [HumanMessage(content=user_input)]},
            config={"recursion_limit": 50}
        )
        return result["messages"][-1]
    except Exception as e:
        return AIMessage(content=f"Error: {str(e)}\n\nPlease try rephrasing your request or provide more specific details.")

if __name__ == "__main__":
    print(
f"""{"=" * 60}
DupeRemove Agent - Duplicated Data Remover
{"=" * 60}
Removes and fixes duplicated data in JSONs.

Examples:
    - Find the duplicated data in data.json and show where it is in the console.
    - Create a new file that removes the duplicated data in users.json.
    - Edit the dupe.json file to remove any repeating data.

Commands: 'quit' or 'exit' to end.
{"=" * 60}"""
    )

    history: List[BaseMessage] = []

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q', ""]:
            print("Goodbye!")
            break

        print("Agent: ", end="", flush=True)
        response = run_agent(user_input, history)
        print(response.content)
        print()

        history += [HumanMessage(content=user_input), response]
