# =========================
# Load Dependencies
# =========================

import os
import hashlib
import requests
from PIL import Image
import streamlit as st
from typing import Set
from io import BytesIO
from dotenv import load_dotenv
from backend.core import run_llm

# =======================================
# Environment and Streamlit Configuration
# =======================================

# Load environment variables from .env file
load_dotenv()

# Configure Streamlit page settings
st.set_page_config(page_title="ChatDocumentation", page_icon="🧊", layout="wide", initial_sidebar_state="expanded")

# Build path to CSS file and apply styles
css_path = os.path.join(os.getcwd(), "static", "css", "styles.css")

# # Open CSS file safely and apply styles
with open(file=css_path, mode="r", encoding="utf-8") as cssfile:
    st.markdown(f"<style>{cssfile.read()}</style>", unsafe_allow_html=True)

# =========================
# Utility Functions
# =========================

def create_sources_string(source_urls: Set[str]) -> str:
    """Generate a formatted string of source URLs."""

    # Format source URLs into a numbered list
    if not source_urls:
        return ""
    
    # Generate the sources string
    return "sources:\n" + "\n".join(f"{i}. {url}" for i, url in enumerate(sorted(source_urls), start=1)) + "\n"

def get_profile_picture(email: str) -> Image.Image:
    """Fetch the Gravatar profile picture for the given email."""

    # Create MD5 hash of the email
    email = email.strip().lower()

    # Generate URL to fetch Gravatar image
    email_hash = hashlib.md5(email.encode("utf-8")).hexdigest()

    # Construct Gravatar URL
    gravatar_url = (f"https://www.gravatar.com/avatar/{email_hash}?d=identicon&s=200")

    # Fetch and return the image
    response = requests.get(gravatar_url, timeout=5)

    # Raise error for bad response
    response.raise_for_status()

    # Load and return image
    return Image.open(BytesIO(response.content))

# =========================
# Sidebar: User Profile
# =========================

with st.sidebar:
    # Display user profile information: image, name, email
    st.title("User Profile")
    st.image(image=get_profile_picture(email="devsofmukesh@gmail.com"), width=150)
    st.write(f"**Name:** Mukesh Kumar")
    st.write(f"**Developer:** devsofmukesh@gmail.com")

# Display main application header
st.header("LangChain Chatbot")

# Initialize session state if missing
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

# =========================
# Layout: Input Section
# =========================

# Create two columns for a more modern layout
column1, column2 = st.columns(spec=[2.5, 1])

# Left column: prompt input
with column1:
    # Render text input field
    prompt = st.text_input("Prompt", placeholder="Enter your message here...", label_visibility="collapsed")

# Right column: submit button
with column2:
    # Render submit button
    if st.button("Submit", key="submit", use_container_width=True):
        prompt = prompt or "Hi there!"

# =========================
# LLM Invocation Logic
# =========================

# Execute only when prompt exists
if prompt:

    # Show spinner during processing
    with st.spinner("Generating response..."):

        # Call LLM backend
        generated_response = run_llm(query=prompt, chat_history=st.session_state["chat_history"])

        # Extract unique source URLs
        sources = set(document.metadata.get("source", "Unknown") for document in generated_response["source_documents"])

        # Format final response with sources
        formatted_response = (f"{generated_response['result']} \n\n {create_sources_string(sources)}")

        # Store user prompt, generated response, and chat history in session state
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["result"]))

# =========================
# Chat History Rendering
# =========================

# Render chat messages if history exists
if st.session_state["chat_answers_history"]:

     # Iterate through conversation history
    for generated_response, user_query in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(generated_response)

# =========================
# Footer
# =========================

# Add footer text
st.markdown("---")
st.markdown("Powered by LangChain and Streamlit")
