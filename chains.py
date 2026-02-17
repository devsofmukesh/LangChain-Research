from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define the prompt template for the agent reflection
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define the prompt template for the agent generation
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            "Generate the best twitter post possible for the user's request."
            "If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Initialize the Ollama LLM
llm = ChatOllama(model="llama3.1:8b")

# The generate_chain is used to generate the agent's reasoning and tool calls based on the input messages
generate_chain = generation_prompt | llm

# Reflect on the agent's reasoning and tool calls to iteratively improve its responses based on the results it has received.
reflect_chain = reflection_prompt | llm
