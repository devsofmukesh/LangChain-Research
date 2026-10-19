# ==========================
# Load Dependencies
# ==========================

from utils import format_box
from dotenv import load_dotenv
from langsmith import traceable
from langchain.tools import tool
from utils import format_box_sections
from langchain_core.messages import ToolMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain.chat_models import init_chat_model


# ==========================
# Load Environment Variables
# ==========================

load_dotenv()


# ==========================
# Constants
# ==========================

MAX_ITERATIONS = 10      # Maximum number of iterations the agent will run before giving up
MODEL = "qwen3.5:9b"     # Change this to your preferred model


# ==========================
# Tools 
# ==========================

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""

    # Simulated product catalog with prices
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}

    # Return the price if found, otherwise return 0
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold.
    """

    # Define discount percentages for each tier
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}

    # Get the discount percentage for the specified tier, defaulting to 0 if not found
    discount = discount_percentages.get(discount_tier, 0)

    # Calculate and return the final price after applying the discount, rounded to 2 decimal places
    return round(price * (1 - discount / 100), 2)


# ==========================
# Main Agent Loop
# ==========================

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    
    # Initialize the LLM and bind tools
    tools = [get_product_price, apply_discount]

    # Create a dictionary for quick tool lookup
    tools_dict = {t.name: t for t in tools}

    # Initialize the LLM with the specified model and temperature
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)

    # Bind the tools to the LLM
    llm_with_tools = llm.bind_tools(tools)

    # Print the question in a formatted box
    print(format_box(f"Question: {question}"))

    # Set up the initial messages for the agent
    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    # Run the agent loop for a maximum number of iterations
    for iteration in range(1, MAX_ITERATIONS + 1):

        # Invoke the LLM with the current messages to get the AI's response
        ai_message = llm_with_tools.invoke(messages)

        # Extract any tool calls from the AI's response
        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(
                format_box_sections(
                    [
                        f"Iteration {iteration}",
                        f"Final Answer: {ai_message.content}",
                    ],
                    width=82,
                )
            )
            return ai_message.content

        # Process only the FIRST tool call — force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        tool_to_use = tools_dict.get(tool_name)

        # If the tool is not found, raise an error and print the iteration details
        if tool_to_use is None:
            error = f"Tool '{tool_name}' not found"
            print(
                format_box_sections(
                    [
                        f"Iteration {iteration}",
                        f"[Tool Selected] {tool_name} with args: {tool_args}",
                        error,
                    ],
                    width=82,
                )
            )
            raise ValueError(error)

        # Invoke the selected tool with the provided arguments and capture the observation
        observation = tool_to_use.invoke(tool_args)

        # Print the iteration details, including the tool selected, its activity, and the observation
        if tool_name == "get_product_price":
            tool_activity = f"Looking up price for product: {tool_args.get('product')}"
        elif tool_name == "apply_discount":
            tool_activity = (f"Applying discount to price: {tool_args.get('price')} tier: {tool_args.get('discount_tier')}")
        else:
            tool_activity = f"Running tool: {tool_name}"

        # Print the iteration details in a formatted box
        print(
            format_box_sections(
                [
                    f"Iteration {iteration}",
                    f"[Tool Selected] {tool_name} with args: {tool_args}",
                    tool_activity,
                    f"[Tool Result] {observation}",
                ],
                width=82,
            )
        )

        # Append the AI's message and the tool's observation to the messages list for the next iteration
        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    # If the maximum number of iterations is reached without a final answer, print an error message
    print(format_box("ERROR: Max iterations reached without a final answer"))
    return None


# ==========================
# Main Execution
# ==========================

if __name__ == "__main__":
    # Print a header for the LangChain Agent Loop and run the agent with a sample question
    print(format_box("LangChain Agent Loop - Shopping Assistant", width=96))

    # Run the agent with a sample question about the price of a laptop after applying a gold discount
    result = run_agent("What is the price of a laptop after applying a gold discount?")
