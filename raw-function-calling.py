# ==========================
# Load Dependencies
# ==========================

import ollama
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
    """Apply a discount tier and return the final price.

    Available tiers: bronze, silver, gold.
    """

    # Define the discount percentages for each tier
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}

    # Use a 0% discount when the requested tier is not recognized
    discount = discount_percentages.get(discount_tier, 0)

    # Return the discounted price rounded to two decimal places
    return round(price * (1 - discount / 100), 2)


# ==========================
# Tool Schemas
# ==========================

# Without LangChain's @tool decorator, define the JSON schemas manually.
# LangChain normally generates these from function type hints and docstrings.
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": (
                            "The product name, e.g. 'laptop', 'headphones', "
                            "'keyboard'"
                        ),
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": (
                "Apply a discount tier to a price and return the final price. "
                "Available tiers: bronze, silver, gold."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The original price",
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": (
                            "The discount tier: 'bronze', 'silver', or 'gold'"
                        ),
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

# Ollama can also generate these schemas when passed the functions directly:
# tools_for_llm = [get_product_price, apply_discount]
# That approach relies on Google-style docstrings to describe each argument.
# The manual schemas above make the JSON sent to the model explicit.


# ==========================
# Traced Ollama Chat Helper
# ==========================

# Without LangChain, trace the Ollama model call directly with LangSmith.
@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    """Call Ollama with the current conversation and available tool schemas."""

    # Pass the tool schemas to Ollama so the model can call them
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


# ==========================
# Main Agent Loop
# ==========================

@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    # Map tool names to Python functions for dispatching model requests
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    # Print the user's question in a formatted box
    print(format_box(f"Question: {question}"))

    # Start the conversation with the assistant's rules and the user question
    messages = [
        {
            "role": "system",
            "content": (
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
            ),
        },
        {"role": "user", "content": question},
    ]

    # Run the agent for at most MAX_ITERATIONS model responses
    for iteration in range(1, MAX_ITERATIONS + 1):

        # Call Ollama directly, then read the assistant's response
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message
        tool_calls = ai_message.tool_calls

        # If there are no tool calls, return the assistant's final answer
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

        # Process only the first tool call in this iteration
        tool_call = tool_calls[0]

        # Ollama exposes the function name and arguments as object attributes
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        # Find and call the requested Python function
        tool_to_use = tools_dict.get(tool_name)
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

        # Describe the selected tool's activity for the iteration output
        if tool_name == "get_product_price":
            tool_activity = (
                f"Looking up price for product: {tool_args.get('product')}"
            )
        elif tool_name == "apply_discount":
            tool_activity = (
                f"Applying discount to price: {tool_args.get('price')} "
                f"tier: {tool_args.get('discount_tier')}"
            )
        else:
            tool_activity = f"Running tool: {tool_name}"

        # Call the function directly with the model-provided arguments
        observation = tool_to_use(**tool_args)

        # Print the iteration details in a single box with dashed separators
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

        # Add the assistant response and tool result for the next model call
        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(observation),
            }
        )

    # Stop if the agent did not produce a final answer within the limit
    print(format_box("ERROR: Max iterations reached without a final answer"))
    return None


# ==========================
# Main Execution
# ==========================

if __name__ == "__main__":

    # Print a header for the shopping assistant
    print(format_box("Raw Function Calling - Shopping Assistant", width=96))

    # Run the agent with a sample question
    result = run_agent(
        question="What is the price of a laptop after applying a gold discount?"
    )
