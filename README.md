# Ask-RAG-LLM

Ask-RAG-LLM is a Streamlit-powered application that combines Retrieval-Augmented Generation (RAG) and Large Language Models (LLM). Users can query predefined topics (e.g., religion, soccer, ADHD help) or upload their own documents to get intelligent, context-aware responses.

## Features :

Predefined Topics: Explore built-in categories like religion, soccer, and ADHD help.
Custom Document Uploads: Upload your own files and query them dynamically.
Top-K Retrieval: Retrieves the top 3 relevant chunks of text from your selected documents.
LLM Integration: Generates insightful answers based on retrieved data.

## How It Works :

1.Preparing the Data - Preloaded or uploaded documents are split into manageable chunks for processing.

2.Creating Chroma Database - Text chunks are stored in a Chroma database for fast similarity searches.

3.Vector Embeddings - Documents and user queries are transformed into vector embeddings using a pre-trained model.

4.Querying for Relevant Data - The system retrieves the top 3 most relevant chunks based on user input.

5.Crafting a Great Response - Relevant chunks are passed to the LLM to generate a detailed and contextually accurate response.



Enjoy discovering knowledge with Ask-RAG-LLM! 🚀

https://sivagugan30-ask-rag-llm.streamlit.app/
