import random
import numpy as np
import streamlit as st
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from openai import OpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import build.custom_functions as cf
import base64
# Set up API Key
os.environ["OPENAI_API_KEY"] = st.secrets["KEY"]

# Predefined list of JSON file paths
json_files = [
    "famous_five_1.json",
    "famous_five_2.json",
    "famous_five_3.json",
    "famous_five_4.json",
    "famous_five_5.json"
]

# Sidebar navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a Section", [
    "Home", "Instructions", "Understand RAG", "Chat-bot", "What's Next?"
])

# Home Section
if options == "Home":
    cf.set_background(image_path='sam_altman.png')
    st.title("Welcome to the Ask-RAG-LLM Application")
    st.markdown(""" """)

# Chat-bot Section
elif options == "Chat-bot":
    # Sample Questions
    st.sidebar.write("### What's Making Waves:")
    st.sidebar.write("1. What does Elon Musk see as the biggest misconception about AI's future, and how will it evolve in the next five years?")
    st.sidebar.write("2. How does Sam Altman view the balance between AI innovation and ethical safeguards in daily life?")
    st.title("RAG Chat-bot")
    
    st.markdown("""
            
            Ask questions related to tech leaders' insights, and get answers generated using RAG with LLM.
            """)



    # Initialize session state for conversation history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # User input for the query
    user_input = st.chat_input("Ask me anything about the documents:")

    if user_input:
        # Add the user input to the conversation history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Process the input to get a response from the model
        vector_dict = cf.load_json_files(json_files)
        query_embeddings = cf.generate_query_embeddings(user_input)
        
        # Retrieve top results based on the query embeddings
        results = cf.query_vector_dict(vector_dict, query_embeddings=query_embeddings, n_results=3)

        # Construct the prompt for the LLM
        prompt = f"""
                    Based on the retrieved documents and user query, generate a response.

                    Query: " {user_input} "

                    Top 3 results:
                    1 >>>>> {results['documents'][0]}
                    2 >>>>> {results['documents'][1]}
                    3 >>>>> {results['documents'][2]}

                    Metadata:
                    - Source:
                        1 >>>>> {results['metadata'][0]['source']}
                        2 >>>>> {results['metadata'][1]['source']}
                        3 >>>>> {results['metadata'][2]['source']}

                    - Start Index:
                        1 >>>>> {results['metadata'][0]['start_index']}
                        2 >>>>> {results['metadata'][1]['start_index']}
                        3 >>>>> {results['metadata'][2]['start_index']}

                    If the context does not provide enough information, respond with "The context does not provide enough information to answer the query."
        """
        
        try:
            # Make the request to OpenAI to get the response
            reply = OpenAI().chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "developer", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt}
                ]
            )

            # Display the response
            assistant_reply = reply.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
        
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Error generating response: {e}"})

    # Display chat history
    for message in st.session_state.messages:
        if message['role'] == "user":
            st.chat_message("user").markdown(message['content'])
        else:
            st.chat_message("assistant").markdown(message['content'])

    
# Instructions Section
elif options == "Instructions":
    st.title("Instructions")
    
    st.write("1. **RAG**: Understand the concept of RAG technique with a sample prompt.")
    st.write("2. **Chat-bot**: Query the embedded documents and retrieve AI-enhanced answers.")
    st.write("3. **What's Next?**: Explore advanced usage suggestions and future updates.")
    
    st.write("")
    st.write("I have tried to make it simple and easy to use. Hope you find the app useful :) ")


# Chat-bot Section
elif options == "Understand RAG":
    #st.title("RAG Chatbot")
    
    st.title("Retrieval-Augmented Generation(RAG)")

    st.write("_RAG improves answers by first retrieving relevant information from a database, then using a model to generate accurate, context-aware responses._") 
    st.write("_RAG is used when the LLM doesn't have enough context on its own._")
    st.write(" ")
    query_text = st.text_input("User Prompt: ", value="How does Satya Nadella differentiate AI agents from traditional software or automation tools?")

    if st.button("Generate Response", key="generate_button", help="Click to initialise the RAG model", use_container_width=True):

        if query_text:
            
                vector_dict1 = cf.load_json_files(json_files)
                
                # Generate embeddings for the query text
                query_embeddings = cf.generate_query_embeddings(query_text)
                
                # Retrieve the top 3 results using the query embeddings
                results = cf.query_vector_dict(
                    vector_dict1, 
                    query_embeddings=query_embeddings,
                    n_results=3
                )
                
                # Construct the prompt for the LLM
                prompt = f"""
                            Basis the retrieved text chunks and the initial user query, generate a response.
                
                            Query: " {query_text} "
                
                            Top 3 results:
                            1 >>>>> {results['documents'][0]}
                            2 >>>>> {results['documents'][1]}
                            3 >>>>> {results['documents'][2]}
                
                            Metadata:
                            - Source:
                                1 >>>>> {results['metadata'][0]['source']}
                                2 >>>>> {results['metadata'][1]['source']}
                                3 >>>>> {results['metadata'][2]['source']}
                
                            - Start Index:
                                1 >>>>> {results['metadata'][0]['start_index']}
                                2 >>>>> {results['metadata'][1]['start_index']}
                                3 >>>>> {results['metadata'][2]['start_index']}
                
                            Mention the Source and Start Index as well seperately in a two new line under 'Source:'. The answer should be structured and simple. 
                
                            If the context does not provide enough information, respond with "The context does not provide enough information to answer the query."
                """
                
                # Make the request to OpenAI to get the response
                try:
                    reply = OpenAI().chat.completions.create(
                        model="gpt-4",  # Fixed model name typo from "gpt-4o" to "gpt-4"
                        messages=[
                            {"role": "developer", "content": "You are a helpful assistant"},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    
                    # Display the response content
                    st.success(reply.choices[0].message.content)
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")
    
    
                st.markdown("###  RAG = Retrive + Augment + Generate ")
                # Display the retrieved results and prompt for transparency
                with st.expander("1. Retrieve", expanded=False):
                    st.write("_Retrieved top 3 results basis Cosine Similarity on user prompt's embeddings and vector database_")
    
                    
                    results1 = {
                                "distances" : results["distances"],
                                "documents" : results["documents"],
                                "metadata"  : results["metadata"],
                            }
                    
                    
                    st.write(results1)
                
                    short_distances = [round(results1["distances"][i], 2) for i in range(3)]
                    short_documents = [
                        results1["documents"][i][:10] + "..." if len(results1["documents"][i]) > 10 else results1["documents"][i]
                        for i in range(3)
                    ]
                    short_metadata = [
                        results1["metadata"][i]["source"][:10] + "..." if len(results1["metadata"][i]["source"]) > 10 else results1["metadata"][i]["source"]
                        for i in range(3)
                    ]
                    
                    # Combine the processed results into the desired output format
                    shortened_results = {
                        "distances": short_distances,
                        "documents": short_documents,
                        "metadata": short_metadata
                    }
    
                # Display the shortened version in Streamlit
                with st.expander("2. Augment", expanded=False):
                    st.code(f"""
                    Augmention = User Prompt + Retrieved Results 
                    """)
                    st.code(f"User query : ' {query_text} ' ")
                    st.code("Retrived Results : ")
                    st.write(shortened_results)  # Display results in JSON-like format
    
                
                with st.expander("3. Generate", expanded=False):
                    st.write("_Augmented prompt is passed to the LLM for generating a response_")
                    st.code(f"Generated response: '{reply.choices[0].message.content}'")
                                    
        else:
            st.warning("Please enter a query to get results")


# What's Next Section
elif options == "What's Next?":
    st.title("What's Next?")
    
    st.write("Explore advanced features and future updates:")
    st.write("- **Advanced Querying:** Enable filtering by metadata or embedding scores.")
    st.write("- **Improved UI/UX:** Design enhancements for a seamless experience.")
    st.write("- **Integration with External APIs:** Add connections to external data sources for richer context.")










