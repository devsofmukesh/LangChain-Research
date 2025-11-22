from typing import List
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.tools import BaseTool
from langchain_ollama import ChatOllama
from callbacks import AgentCallbackHandler
from langchain_core.messages import ToolMessage
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Decorator converting Python function into LangChain tool
@tool
def get_text_length(text: str) -> str:
    """Returns the length of the text by characters."""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip('"')
    return len(text)

def find_tool_by_name(tools: List[BaseTool], tool_name: str) -> BaseTool:
    """Returns the tool from the list that matches the given name."""
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool wtih name {tool_name} not found")


if __name__ == "__main__":
    # Display startup message
    print("Hello LangChain Tools (.bind_tools)!")
    
    # Define the list of available tools
    tools = [get_text_length]

    # Initialize the Ollama chat model with configuration
    llm = ChatOllama(
        model="llama3.1:8b",
        temperature=0.0,
        stop=["\nObservation", "Observation", "Observation:"],
        callbacks=[AgentCallbackHandler()]
    )

    # Bind the tools to the LLM for tool calling
    llm_with_tools = llm.bind_tools(tools)

    # Start conversation
    messages = [HumanMessage(content="What is the length of the word: DOG")]

    while True:
        # Invoke the model with the current conversation history
        ai_message = llm_with_tools.invoke(messages)

        # Retrieve any tool calls the model wants to make
        tool_calls = getattr(ai_message, "tool_calls", None) or []
        if len(tool_calls) > 0:
            # Store the model message that requested tool calls
            messages.append(ai_message)
            for tool_call in tool_calls:
                # Extract the tool name, arguments, and call ID
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id")
                
                # Locate the appropriate tool in the list of tools
                tool_to_use = find_tool_by_name(tools, tool_name)
                observation = tool_to_use.invoke(tool_args)
                print(f"observation={observation}")
                # Add the tool's response back into the message history
                messages.append(
                    ToolMessage(content=str(observation), tool_call_id=tool_call_id)
                )
            # Allow the model to process the tool results
            continue

        # If no tool calls are present, treat this as the final answer and exit the loop
        print(ai_message.content)
        break
