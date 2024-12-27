import os 
import streamlit as st
from typing import List
import langchain




os.environ["OPENAI_API_KEY"] = st.secrets["KEY"]

# App title
st.title("Ask-RAG-LLM")

# Sidebar for topic selection and document upload
st.sidebar.header("Options")
topics = ["Religion", "Soccer", "ADHD Help"]
selected_topic = st.sidebar.selectbox("Choose a Topic", topics)

uploaded_file = st.sidebar.file_uploader("Upload a Document (Optional)", type=["txt", "pdf"])

# Preload or upload documents
if uploaded_file:
    st.sidebar.write("✅ File uploaded successfully!")
else:
    st.sidebar.write("📚 Using predefined topic:", selected_topic)

# User query input
query = st.text_input("Ask your question:", placeholder="Type your question here...")

# Submit button
if st.button("Get Answer"):
    if not query:
        st.warning("Please enter a question!")
    else:
        # Simulated workflow
        st.write("### Top 3 Retrieved Sources")
        # Simulated sources (replace with retrieval logic)
        sources = [
            "Source 1: Relevant text from the selected topic or document...",
            "Source 2: Another piece of relevant text...",
            "Source 3: Yet another relevant text snippet..."
        ]
        for i, source in enumerate(sources, start=1):
            st.write(f"**Source {i}:** {source}")

        # Simulated LLM response (replace with LLM call)
        st.write("### LLM Response")
        st.success("This is a simulated response generated by the LLM based on the top sources.")

# Footer
st.sidebar.markdown("----")
st.sidebar.caption("Powered by Streamlit, RAG, and LLM")
