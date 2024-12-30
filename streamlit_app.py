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
import faiss

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
    "Home", "Instructions", "Document Embedding", "Chat-bot", "What's Next?"
])

# Home Section
if options == "Home":
    st.title("Welcome to the Ask-RAG-LLM Application")
    st.markdown(""" """)

# Instructions Section
elif options == "Instructions":
    st.title("Instructions")
    
    st.write("1. **Document Embedding**: Upload or select text for vector database generation.")
    st.write("2. **Chat-bot**: Query the embedded documents and retrieve AI-enhanced answers.")
    st.write("3. **What's Next?**: Explore advanced usage suggestions and future updates.")
    
    st.write("")
    st.write("I have tried to make it simple and easy to use. Hope you find the app useful :) ")


# Document Embedding Section
elif options == "Document Embedding":
    st.title("Document Embedding")
    st.markdown("<p style='font-size: 30px;'>Convert <strong>Text</strong> to <strong>Numerical Representation</strong></p>", unsafe_allow_html=True)

    with st.form("document_input"):
        # Use st.sidebar.selectbox instead of st.radio
        input_type = st.sidebar.selectbox("Choose Input Type", ["Select from Existing Documents", "Upload Markdown File", "Paste Text"])
        documents = []

        if input_type == "Select from Existing Documents":
            existing_documents = ["Doc 1", "Doc 2", "Doc 3"]  # Replace with actual documents
            selected_doc = st.selectbox("Select a Document", existing_documents)
            documents = [Document(page_content=f"Content of {selected_doc}", metadata={"source": selected_doc})]

        elif input_type == "Upload Markdown File":
            uploaded_files = st.file_uploader("Upload Markdown Files", type=["md"], accept_multiple_files=True)
            if uploaded_files:
                for file in uploaded_files:
                    content = file.read().decode("utf-8")
                    metadata_source = file.name
                    documents.append(Document(page_content=content, metadata={"source": metadata_source}))
                st.write(f"Uploaded {len(documents)} documents.")

        elif input_type == "Paste Text":
            pasted_text = st.text_area("Paste your text here (max 500 words)", height=200)
            if len(pasted_text.split()) > 500:
                st.warning("Text exceeds 500 words. Please limit your input to 500 words.")
            elif pasted_text:
                documents = [Document(page_content=pasted_text, metadata={"source": "Pasted Text"})]

        row_1 = st.columns([2, 1])
        with row_1[0]:
            chunk_size = st.number_input("Chunk Size", value=300, min_value=0, step=1, help="Maximum characters in each chunk")

        with row_1[1]:
            chunk_overlap = st.number_input("Chunk Overlap", value=100, min_value=0, step=1, help="Overlap between chunks")

        row_2 = st.columns([2, 1])
        with row_2[0]:
            source_name = st.text_input("New Vector Store Name", "vector_store_1", help="Name for the vector store")

        with row_2[1]:
            add_start_index = st.selectbox("Add Start Index", [True, False], index=0, help="Include start index in chunks")

        save_button = st.form_submit_button("Generate Vector DB")

        if save_button:
            if documents:
                chunks = cf.split_text(documents, chunk_size, chunk_overlap, add_start_index)
                if chunks:
                    random_index = random.randint(0, len(chunks) - 1)
                    st.write(f"Randomly selected chunk (index {random_index}):")
                    st.write("Metadata:", chunks[random_index].metadata)
                    st.write({"text": chunks[random_index].page_content})
            else:
                st.warning("Please upload or provide text input.")


# Chat-bot Section
elif options == "Chat-bot":
    st.title("RAG Chatbot")
    
    st.markdown("### Retrieval-Augmented Generation (RAG)")
    
    query_text = st.text_input("Enter your query: ", value="What is the name of the island?")

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
    
    
                st.write("  RAG = Retrive + Augment + Generate ")
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










