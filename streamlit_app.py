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
    "tech_1.json",
    "tech_2.json",
    "tech_3.json",
    "tech_4.json",
    "tech_5.json",
    "tech_6.json",
    "tech_7.json",
    "tech_8.json"
]

# Sidebar navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a Section", [
    "Home", "Instructions", "Understand RAG", "Chatbot", "What's Next?"
])

# Home Section
if options == "Home":

    st.title("Welcome to RAG-based Chatbot!")
    st.write("")
    st.write("Hello! I’m Siva, and I love building data products that make a difference.")
    st.write("")
    st.markdown("\t _'Simplicity is the ultimate sophistication' - Da Vinci_  ")
    st.markdown("\t _'Simple can be harder than complex' - Steve Jobs_")
    st.write("")
    
    # Add your own statement
    st.markdown("I've tried to make the app simple and easy to use. Hope you find it useful :)")
    st.write("")
    st.write("")
    st.write("")
    

    st.write("**Next Page** : Use the Navigation bar in the top left corner to start")



# Chat-bot Section
elif options == "Chatbot":
    # Sample Questions
    st.sidebar.write("### Trending Now:")
    st.sidebar.write("1. What is Mark Zuckerberg’s take on balancing open-source with global competition?")
    st.sidebar.write("2. How does Sam Altman view the balance between AI innovation and ethical safeguards in daily life?")
    st.sidebar.write("3. What does Jensen Huang think is the next big leap for GPUs in advancing AI?")
    st.title("RAG-Chatbot")
    
    st.markdown("""
            
            Ask questions on Tech leaders' opinions extracted from podcasts and receive answers generated using RAG with LLM
            
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
                    1 - {results['documents'][0]}
                    2 - {results['documents'][1]}
                    3 - {results['documents'][2]}

                    Metadata(youtube details):
                        1 - youtube channel : {results['metadata'][0]['video_channel']} | youtube link : {results['metadata'][0]['video_url']}
                        2 - youtube channel : {results['metadata'][1]['video_channel']} | youtube link : {results['metadata'][1]['video_url']}
                        3 - youtube channel : {results['metadata'][2]['video_channel']} | youtube link : {results['metadata'][2]['video_url']}
                        
                    If the context does not provide enough information, reply by saying : Please note that the current sources available to RAG are limited to 8 YouTube podcasts of Tech leaders, so there may not be specific information related to your query. Apologies   """
        
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
    
    st.write("1. **Understand RAG**: Learn how RAG works with a simple example")
    st.write("2. **Chat-bot**: Ask questions and get answers based on YouTube podcasts of Tech leaders")
    st.write("3. **What's Next?**: Explore how the app will improve and offer more features in the future")

# Chat-bot Section
elif options == "Understand RAG":
    #st.title("RAG Chatbot")
    
    st.title("Retrieval-Augmented Generation(RAG)")

    st.write("• _RAG is used when LLMs (Large Language Models) lack detailed knowledge about a specific topic or domain._")
    st.write("• _RAG improves answers by first retrieving relevant information from an external database, then using LLM to generate accurate, context-aware responses._")
 
    st.write(" ")
    query_text = st.text_input("User Prompt: ", value="How does Satya Nadella view the impact of Agent AI?")

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
                
                            Metadata(youtube details):
                            1 - youtube channel : {results['metadata'][0]['video_channel']} | youtube link : {results['metadata'][0]['video_url']}
                            2 - youtube channel : {results['metadata'][1]['video_channel']} | youtube link : {results['metadata'][1]['video_url']}
                            3 - youtube channel : {results['metadata'][2]['video_channel']} | youtube link : {results['metadata'][2]['video_url']}
                        
                    'If the context does not provide enough information, reply by saying : Please note that the current sources available to RAG are limited to 8 YouTube podcasts of Tech leaders, so there may not be specific information related to your query. Apologies'  
                
                            Mention the Source and Start Index as well seperately in a two new line under 'Source:'. The answer should be structured and simple. 
                            
                            Provide a concise answer within 50 words, including 'sources' (list only the YouTube video title and link on a new line).
                            
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
    
                    st.write("_First, Cosine Similarity is applied to the user prompt's embeddings and the vector database to **RETRIEVE** the most relevant results._")
                    
                    results1 = {
                                "distances" : results["distances"],
                                "documents" : results["documents"],
                                "metadata"  : results["metadata"],
                            }
                    
                    st.write(results1)
                
                    
    
                # Display the shortened version in Streamlit
                with st.expander("2. Augment", expanded=False):

                    short_distances = [round(results["distances"][i], 2) for i in range(3)]
                    short_documents = [
                        results["documents"][i][:35] + "..." if len(results["documents"][i]) > 35 else results["documents"][i]
                        for i in range(3)
                    ]
                    short_metadata = [
                        f"{results['metadata'][i]['speaker'][:10]} | " +
                        f"{results['metadata'][i]['video_name'][:10]}... | " +
                        f"{results['metadata'][i]['video_channel'][:10]}... | " +
                        f"{results['metadata'][i]['date']} | " +
                        f"{results['metadata'][i]['video_timestamp']} | " +
                        f"{results['metadata'][i]['video_url'][:25]}..."
                        for i in range(3)
                    ]
                    
                    # Combine the processed results into the desired output format
                    shortened_results = {
                        "distances": short_distances,
                        "documents": short_documents,
                        "metadata": short_metadata
                    }
                    
                    prompt1 = f"""
                                Hey LLM, below is the user query and the relevant results. Paraphrase a response.
                    
                                Query: " {query_text} "
                                
                                Top 3 results: \n
                                \t 1 : {shortened_results['documents'][0]} | d = {shortened_results['distances'][0]} \n
                                \t 2 : {shortened_results['documents'][1]} | d = {shortened_results['distances'][1]} \n
                                \t 3 : {shortened_results['documents'][2]} | d = {shortened_results['distances'][2]} \n
                            
                                Metadata(source): \n
                                \t 1 : {shortened_results['metadata'][0]} \n
                                \t 2 : {shortened_results['metadata'][1]} \n
                                \t 3 : {shortened_results['metadata'][2]}
                            """
                    
                                        
                    
                    st.write("_Instead of feeding just the prompt to the LLM, we **AUGMENT** the prompt by adding retrieved results for better response generation._")
                    
                    st.code('Augmented Prompt (redacted version): ')
                    st.info(prompt1)
                
                
                with st.expander("3. Generate", expanded=False):
                    st.write("_Finally, the augmented prompt (user prompt + results) is fed to the LLM to **GENERATE** a response._")
                    st.code(f"Generated response: '{reply.choices[0].message.content}'")
                                    
        else:
            st.warning("Please enter a query to get results")


elif options == "What's Next?":
    st.title("The Road Ahead:")
    st.write("")
    st.write("While current baseline RAG model has it's limitations in connecting concepts and summarizing effectively, I’m happy to have learned a new concept")
    st.write("")
    st.write("By integrating Graph RAG, I aim to enhance the model's ability to establish relationships and offer deeper, context-rich insights for complex domains")
    st.write("")
    st.write("Also, my friends have asked for the option to upload their own documents to help with their academic and research works")
    st.write("")
    st.write("Hence, the way forward is to allow users to upload documents or paste YouTube links, enabling the Graph-RAG-based app to better serve their needs.")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Created by Sivagugan Jayachandran")
