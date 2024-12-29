import random
import numpy as np
import streamlit as st
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
from openai import OpenAI
import os 
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import build.custom_functions as cf

os.environ["OPENAI_API_KEY"] = st.secrets["KEY"]

# Predefined list of JSON file paths
json_files = [
    "famous_five_1.json",
    "famous_five_2.json",
    "famous_five_3.json",
    "famous_five_4.json",
    "famous_five_5.json"
]

# Tabs
tabs = st.tabs(["Document Embedding", "RAG Chatbot"])


with tabs[0]:
    
    st.markdown("<p style='font-size: 30px;'>Convert <strong>Text</strong> to <strong>Numerical Representation</strong></p>", unsafe_allow_html=True)
    
    with st.form("document_input"):
        # Dropdown to choose between existing documents, uploading a file, or pasting text
        input_type = st.radio("Choose Input Type", ["Select from Existing Documents", "Upload Markdown File", "Paste Text"])

        documents = []  # List to hold the documents

        if input_type == "Select from Existing Documents":
            # Example: List of pre-existing documents (this can be replaced by your actual source of documents)
            existing_documents = ["Doc 1", "Doc 2", "Doc 3"]  # Replace with your actual document list
            selected_doc = st.selectbox("Select a Document", existing_documents)

            # Add selected document content (for demo, using placeholder text)
            documents = [Document(page_content=f"Content of {selected_doc}", metadata={"source": selected_doc})]

        elif input_type == "Upload Markdown File":
            uploaded_files = st.file_uploader(
                "Upload Markdown Files", type=["md"], accept_multiple_files=True
            )
            if uploaded_files:
                for file in uploaded_files:
                    content = file.read().decode("utf-8")
                    metadata_source = source_name if source_name.strip() else file.name
                    documents.append(Document(page_content=content, metadata={"source": metadata_source}))
                st.write(f"Uploaded {len(documents)} documents.")

        elif input_type == "Paste Text":
            pasted_text = st.text_area("Paste your text here (max 500 words)", height=200)
            if len(pasted_text.split()) > 500:
                st.warning("Text exceeds 500 words. Please limit your input to 500 words.")
            elif pasted_text:
                # If the user pastes text, treat it as a document
                documents = [Document(page_content=pasted_text, metadata={"source": "Pasted Text"})]

        # Text chunking options
        row_1 = st.columns([2, 1])
    
        row_1 = st.columns([2, 1])
        with row_1[0]:
            chunk_size = st.number_input(
                "Chunk Size", value=300, min_value=0, step=1,
                help="Specifies the maximum number of characters in each chunk of text"
            )
    
        with row_1[1]:
            chunk_overlap = st.number_input(
                "Chunk Overlap", value=100, min_value=0, step=1,
                help="Defines how much overlap (in characters) exists between consecutive chunks"
            )
    
        row_2 = st.columns([2, 1])
        with row_2[0]:
            source_name = st.text_input(
                "New Vector Store Name", "vector_store_1",
                help="If left empty, the uploaded file's name will be used"
            )
    
        with row_2[1]:
            add_start_index = st.selectbox(
                "Add Start Index", [True, False], index=0,
                help="Choose whether to add the start index to each chunk"
            )
    
        save_button = st.form_submit_button("Generate Vector DB ")
    
        if save_button:
            if uploaded_files:
                documents = []
                for file in uploaded_files:
                    content = file.read().decode("utf-8")
                    metadata_source = source_name if source_name.strip() else file.name
                    documents.append(Document(page_content=content, metadata={"source": metadata_source}))
        
                st.write(f"Uploaded {len(documents)} documents.")
        
                chunks = cf.split_text(documents, chunk_size, chunk_overlap, add_start_index)
        
                
    
                if chunks:
                    # Pick a random chunk index
                    random_index = random.randint(0, len(chunks) - 1)
                    
                    st.write(f"(randomly selected, index {random_index}):")
                    st.write("Metadata:", chunks[random_index].metadata)
                    st.write({"text": chunks[random_index].page_content})
                    
        else:
            st.warning("Please upload at least one Markdown file.")



with tabs[1]:
    
    
    st.markdown("### Retrieval-Augmented Generation (RAG) ")
    
    # Load data from JSON files
    json_files = [
        "famous_five_1.json",
        "famous_five_2.json",
        "famous_five_3.json",
        "famous_five_4.json",
        "famous_five_5.json"
    ]
    
    vector_dict = cf.load_json_files(json_files)
    
    # Query input with default value
    query_text = st.text_input("Enter your query: ", value="What is the name of the island?")
    
    # If the button is clicked, process the RAG response
    if st.button("Generate Response"):
        if query_text:
            
            # Generate embeddings for the query text
            query_embeddings = cf.generate_query_embeddings(query_text)
            
            # Retrieve the top 3 results using the query embeddings
            results = cf.query_vector_dict(
                vector_dict, 
                query_embeddings=query_embeddings,
                n_results=3
                # ,where=where
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
                        {"role": "developer", "content": "You are a helpful assistant."},
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
            st.warning("Please enter a query to get results.")
                                
      
