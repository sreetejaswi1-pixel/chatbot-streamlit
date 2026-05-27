import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Title
st.title("🤖 RAG Chatbot")
st.subheader("Retrieval Augmented Generation")

# Introduction
st.write("""
Welcome to the AI-powered RAG Chatbot 🚀
Ask questions related to AI, ML, and RAG.
""")

# Features
st.header("✨ Features")

st.markdown("""
- Artificial Intelligence chatbot
- Machine Learning concepts
- RAG-based responses
- Streamlit UI
- Interactive chatbot
""")

# User Input
user_input = st.text_input("💬 Ask something:")

# Submit Button
if st.button("Submit"):

    # Check empty input
    if user_input == "":
        st.error("Please enter a question.")

    else:

        st.write("### You asked:")
        st.write(user_input)

        question = user_input.lower()

        # AI Response
        if "artificial intelligence" in question or "ai" in question:

            st.success("""
Artificial Intelligence (AI) is the simulation of human intelligence by machines.
AI enables computers to think, learn, and make decisions.
""")

        # Machine Learning Response
        elif "machine learning" in question or "ml" in question:

            st.success("""
Machine Learning (ML) is a subset of Artificial Intelligence.
It allows systems to learn automatically from data.
""")

        # RAG Response
        elif "rag" in question:

            st.success("""
RAG stands for Retrieval Augmented Generation.

RAG combines:
1. Information Retrieval
2. AI Text Generation

It helps chatbots answer questions using external documents and knowledge sources.
""")

        # Default Response
        else:

            st.info("Sorry, answer not available yet 🚀")

# Footer
st.markdown("---")
st.caption("Built using Streamlit + Python")