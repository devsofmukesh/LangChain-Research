# ==========================
# Load Dependencies
# ==========================

import re
import ollama
import inspect
from utils import format_box
from dotenv import load_dotenv
from langsmith import traceable
from utils import format_box_sections


# ==========================
# Load Environment Variables
# ==========================

load_dotenv()


# ==========================
# Constants
# ==========================

MAX_ITERATIONS = 10         # Maximum number of times the agent can call a tool
MODEL = "qwen3.5:9b"        # Change this to your preferred model


# ==========================
# Tools
# ==========================

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""

    # Simulated product catalog with prices
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}

    # Return the product price, or 0 when the product is not in the catalog
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold.
    """

    # Ensure the value parsed from the model is numeric
    price = float(price)

    # Define the discount percentages for each tier
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)

    # Return the discounted price rounded to two decimal places
    return round(price * (1 - discount / 100), 2)


# Map the names that can appear in the ReAct prompt to their Python functions
tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}


# ==========================
# ReAct Prompt
# ==========================

def get_tool_descriptions(tools_dict):
    """Build prompt descriptions from each tool's signature and docstring."""

    descriptions = []
    for tool_name, tool_function in tools_dict.items():

        # Inspect the undecorated function to avoid wrapper-specific parameters
        original_function = getattr(tool_function, "__wrapped__", tool_function)

        # Get the function signature and docstring for the prompt description
        signature = inspect.signature(original_function)

        # Get the docstring, or use an empty string if none is provided
        docstring = inspect.getdoc(tool_function) or ""

        # Append the formatted description to the list
        descriptions.append(f"{tool_name}{signature} - {docstring}")

    # Return the joined descriptions as a single string
    return "\n".join(descriptions)


# Describe the available tools and list the action names for the prompt
tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

# The model returns plain text in this format; the agent parses Action fields
# from the response instead of sending structured tool schemas to Ollama.
react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""


# ==========================
# Traced Ollama Chat Helper
# ==========================

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    """Send the prompt to Ollama and return its response."""

    return ollama.chat(model=model, messages=messages, options=options)


# ==========================
# Main Agent Loop
# ==========================

@traceable(name="Ollama Raw Agent Loop")
def run_agent(question: str):
    # Print the user's question in a formatted box
    print(format_box(f"Question: {question}"))

    # Create the initial ReAct prompt and an empty history for tool observations
    prompt = react_prompt.format(question=question)
    scratchpad = ""

    # Run the agent for at most MAX_ITERATIONS model responses
    for iteration in range(1, MAX_ITERATIONS + 1):
        full_prompt = prompt + scratchpad

        # Ask the model to continue the ReAct trace. The stop sequence leaves
        # Observation for the agent to fill with the real tool result.
        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation"], "temperature": 0},
        )
        output = response.message.content

        # If the model returned a final answer, display it and finish
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(
                format_box_sections(
                    [
                        f"Iteration {iteration}",
                        f"LLM Output:\n{output}",
                        f"Final Answer: {final_answer}",
                    ],
                    width=82,
                )
            )
            return final_answer

        # Parse the action and its comma-separated input from the model output
        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        # If either the action or its input could not be parsed, display an error and stop
        if not action_match or not action_input_match:
            print(
                format_box_sections(
                    [
                        f"Iteration {iteration}",
                        f"LLM Output:\n{output}",
                        "[Parsing Error] Could not parse Action and Action Input",
                    ],
                    width=82,
                )
            )
            break

        # Extract the tool name and its raw input, then clean and split the arguments
        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        # Split comma-separated values and remove optional key= prefixes
        raw_args = [value.strip() for value in tool_input_raw.split(",")]
        args = [
            value.split("=", 1)[-1].strip().strip("'\"")
            for value in raw_args
        ]

        # Run the requested tool, or return an error as its observation
        if tool_name not in tools:
            observation = (
                f"Error: Tool '{tool_name}' not found. "
                f"Available tools: {list(tools.keys())}"
            )
        else:
            observation = str(tools[tool_name](*args))

        # Display the model output, selected action, and result in one box
        print(
            format_box_sections(
                [
                    f"Iteration {iteration}",
                    f"LLM Output:\n{output}",
                    f"[Tool Selected] {tool_name} with args: {tool_input_raw}",
                    f"[Tool Executing] {tool_name}({args})",
                    f"[Tool Result] {observation}",
                ],
                width=82,
            )
        )

        # Add the model's reasoning and the actual result for the next iteration
        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    # Stop if no final answer was produced within the iteration limit
    print(format_box("ERROR: Max iterations reached without a final answer"))
    return None


# ==========================
# Main Execution
# ==========================

if __name__ == "__main__":
    # Print a header for the agent and run it with a sample question
    print(format_box("Raw ReAct Agent - Shopping Assistant", width=96))

    # Run the agent with a sample question and print the final answer
    result = run_agent(question="What is the price of a laptop after applying a gold discount?")
