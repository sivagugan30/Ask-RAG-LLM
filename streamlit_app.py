import random
import numpy as np
import streamlit as st
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from openai import OpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    "Home", "Instructions", "How RAG works?", "Chatbot", "What's Next?"
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
    st.write('**Last update** : Dec 31, 2024')


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
    user_input = st.chat_input("Ask me anything about the latest trends in Tech!")

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
                            1 - youtube channel : {results['metadata'][0]['video_channel']} | youtube link : {results['metadata'][0]['video_url']} | youtube video name : {results['metadata'][0]['video_name']}
                            2 - youtube channel : {results['metadata'][1]['video_channel']} | youtube link : {results['metadata'][1]['video_url']} | youtube video name : {results['metadata'][1]['video_name']}
                            3 - youtube channel : {results['metadata'][2]['video_channel']} | youtube link : {results['metadata'][2]['video_url']} | youtube video name : {results['metadata'][2]['video_name']}
                        
                   Provide a structured response, including 'sources' with only the YouTube video title and link. Please present the link as a hyperlinked text with the YouTube video title on a new line.

                   If the context does not provide enough information, reply by saying : 'Please note that the current sources available to RAG are limited to 8 YouTube podcasts of Tech leaders, so there may not be specific information related to your query. Apologies'  
                """
        
        try:
            # Make the request to OpenAI to get the response
            reply = OpenAI().chat.completions.create(
                model="gpt-3.5-turbo",
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
    
    st.write("1. **How RAG works?**: Learn how RAG works with a simple example")
    st.write("2. **Chat-bot**: Ask questions and get answers based on YouTube podcasts of Tech leaders")
    st.write("3. **What's Next?**: Explore how the app will improve and offer more features in the future")

# Chat-bot Section
elif options == "How RAG works?":

    st.title("Retrieval-Augmented Generation(RAG)")

    st.write("• _RAG is used when LLMs (Large Language Models) lack detailed knowledge about a specific topic or domain._")
    st.write("• _RAG improves answers by first retrieving relevant information from an external database, then using LLM to generate accurate, context-aware responses._")

    st.write(" ")
    query_text = "How does Satya Nadella view the impact of Agent AI?"
    st.text_input("User Prompt: ", value=query_text, disabled=True)

    # --- Frozen demo data (pre-computed, no API calls) ---
    results = {
        "distances": [0.6372312578428785, 0.6299721035234419, 0.6196395966320398],
        "documents": [
            "[Satya Nadella] : 'there will be disruption but so the way we are approaching at least our M365 stuff is one is build co-pilot as that organizing layer UI for AI get all AG agents including our own agents you can say the Excel is an agent to my co-pilot word is an agent it's kind of specialized canvases which is I'm doing a legal document let me take it into pages and then to word and then have the co-pilot Go With It uh go into Excel and have the co-pilot go with it and so that's sort of a new way to think about'",
            "[Satya Nadella] : 'cloud um and I think in a world where every AI application is a stateful application it's an agentic application that agent agent performs actions then classic app server plus the AI app server plus the data base are all required and so I go back to my fundamental thing which is hey we built this 60 plus AI regions I mean Azure regions they all will be ready for fullon AI applications and that's I think what will be needed that makes it you know that makes a lot of sense um so let's talk a little bit you know we've'",
            "[Satya Nadella] : 'try and collapse it all right whether it's in customer service whether it is in you know uh by the way the other fascinating thing that's increasing is just not CRM but even our what we call finance and operations uh because people want more AI native Biz apps right that means The Biz app the logic tier can be orchestrated by Ai and AI agent so in other words copilot to agent to my business application should be very seamless now in the same way you could even say hey why do I need Excel like interestingly enough one of the most'",
        ],
        "metadata": [
            {"speaker": "Satya Nadella", "video_name": "Satya Nadella | BG2 w/ Bill Gurley & Brad Gerstner", "video_channel": "Bg2 Pod", "date": "12 Dec 2024", "video_timestamp": "00:49:46", "video_url": "https://www.youtube.com/watch?v=9NtsnzRFJ_o"},
            {"speaker": "Satya Nadella", "video_name": "Satya Nadella | BG2 w/ Bill Gurley & Brad Gerstner", "video_channel": "Bg2 Pod", "date": "12 Dec 2024", "video_timestamp": "01:12:09", "video_url": "https://www.youtube.com/watch?v=9NtsnzRFJ_o"},
            {"speaker": "Satya Nadella", "video_name": "Satya Nadella | BG2 w/ Bill Gurley & Brad Gerstner", "video_channel": "Bg2 Pod", "date": "12 Dec 2024", "video_timestamp": "00:47:58", "video_url": "https://www.youtube.com/watch?v=9NtsnzRFJ_o"},
        ],
    }

    reply_text = (
        "Satya Nadella views the impact of Agent AI as a transformative tool for enhancing "
        "productivity and seamless integration with various applications like Excel and Word. "
        "He emphasizes the importance of AI-native business applications and the orchestration "
        "of logic tiers through AI agents. The vision is to enable a smooth transition from "
        "co-pilot to agent to business applications.\n\n"
        "Sources:\n"
        "- [Satya Nadella | BG2 w/ Bill Gurley & Brad Gerstner]"
        "(https://www.youtube.com/watch?v=9NtsnzRFJ_o)"
    )

    st.success(reply_text)

    st.markdown("###  RAG = Retrieve + Augment + Generate ")

    with st.expander("1. Retrieve", expanded=False):

        st.write("_First, Cosine Similarity is applied to the embeddings of user prompt and the external database to **RETRIEVE** the top 3 relevant results._")

        st.code(" 'distances' : Cosine similarity score [0 to 1], where 1 means a perfect match. Higher values indicate more relevance")
        st.code(" 'documents' : Relevant text content from the external database")
        st.code(" 'metadata' : Extra details about the documents, such as their source")

        st.write('Retrived results : ')

        st.write({
            "distances": results["distances"],
            "documents": results["documents"],
            "metadata": results["metadata"],
        })

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
        st.code(f"Generated response: '{reply_text}'")


elif options == "What's Next?":
    st.title("The Road Ahead:")
    st.write("")
    st.write("While the current baseline RAG model has limitations in connecting concepts and summarizing effectively, I’m excited to have learned a new concept.")
    st.write("")
    st.write("By integrating Graph RAG, I aim to enhance the model's ability to establish relationships and offer deeper, context-rich insights for complex domains.")
    st.write("")
    st.write("Upon requests from my friends, I plan to enable the option to upload documents or paste YouTube links for academic and research purposes, allowing the Graph-RAG-based app to better meet everyone' needs.")
    st.write("")
    st.write("Ultimately, my goal is to build an **AI agent** that assists users by processing diverse inputs and providing tailored, context-driven insights.")
    st.write("")
    st.write("Thanks :)")
# Footer
st.sidebar.markdown("---")
st.sidebar.write("Created by Sivagugan Jayachandran")
