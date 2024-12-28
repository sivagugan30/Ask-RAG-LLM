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

os.environ["OPENAI_API_KEY"] = st.secrets["KEY"]


# Predefined list of JSON file paths
json_files = [
    "famous_five_1.json",
    "famous_five_2.json",
    "famous_five_3.json",
    "famous_five_4.json",
    "famous_five_5.json"
]

# Function to load and process JSON data from files
def load_json_files(json_files):
    ids = []
    documents = []
    metadata = []
    embeddings = []
    
    for json_file in json_files:
        #st.write(f"Loading JSON data from: {json_file}")
        try:
            # Open and read the JSON file
            with open(json_file, 'r') as file:
                json_data = json.load(file)  # Parse the JSON content
            #st.write(f"Successfully loaded JSON data from {json_file}")
        except Exception as e:
            st.write(f"Failed to load JSON data from {json_file}: {e}")
            continue  # Skip this file if reading fails

        # Process the data from the JSON
        try:
            #st.write("Processing JSON data.")
            ids.extend(json_data["ids"])  # Append data to the existing list
            documents.extend(json_data["documents"])
            metadata.extend(json_data["metadata"])
            embeddings.extend(json_data["embeddings"])
            #st.write(f"Processed data with {len(json_data['ids'])} rows successfully.")
        except KeyError as e:
            st.write(f"Error: Key {e} not found in the JSON data.")
        except Exception as e:
            st.write(f"Error processing the JSON data: {e}")

    # Convert embeddings to a NumPy array
    embeddings_array = np.array(embeddings)

    # Combine all data into a dictionary
    vector_dict = {
        "ids": ids,
        "documents": documents,
        "metadata": metadata,
        "embeddings": embeddings_array
    }


    return vector_dict


# Function to generate embeddings for a query using OpenAI API
def generate_query_embeddings(query_text):
    query_embeddings = OpenAI().embeddings.create(
        input=query_text,
        model="text-embedding-3-small"  # Specify the embedding model
    ).data[0].embedding

    query_embeddings = np.array(query_embeddings).reshape(1, -1)
    return query_embeddings


# Define the query_vector_dict function
def query_vector_dict(vector_dict, query_texts=None, query_embeddings=None, n_results=3, where=None, where_document=None, include=["metadatas", "documents", "distances"]):
    """
    Query the vector_dict to find the closest neighbors.
    """
    ids = vector_dict["ids"]
    documents = vector_dict["documents"]
    metadata = vector_dict["metadata"]
    embeddings = vector_dict["embeddings"]

    # Function to filter metadata or documents based on where conditions
    def apply_filter(data, filter_condition):
        if filter_condition is None:
            return data
        filtered_data = []
        for item in data:
            if all(item.get(key) == value for key, value in filter_condition.items()):
                filtered_data.append(item)
        return filtered_data

    # Apply the `where` and `where_document` filters
    if where:
        metadata = apply_filter(metadata, where)
    if where_document:
        documents = apply_filter(documents, where_document)

    # Ensure we also filter embeddings and ids based on the metadata or documents filter
    # We need to ensure the filtered metadata is indexed correctly
    filtered_metadata = [metadata[i] for i in range(len(metadata)) if metadata[i] in metadata]
    filtered_ids = [ids[i] for i in range(len(ids)) if metadata[i] in filtered_metadata]
    filtered_documents = [documents[i] for i in range(len(documents)) if metadata[i] in filtered_metadata]
    filtered_embeddings = [embeddings[i] for i in range(len(embeddings)) if metadata[i] in filtered_metadata]

    # Calculate the cosine similarity for query_embeddings or query_texts
    if query_embeddings is not None:
        similarities = cosine_similarity(query_embeddings, filtered_embeddings)
    elif query_texts is not None:
        # Generate embeddings for the query_texts
        query_embeddings = generate_embeddings(query_texts)
        similarities = cosine_similarity(query_embeddings, filtered_embeddings)
    else:
        raise ValueError("Either query_embeddings or query_texts must be provided.")

    # Get the closest neighbors (sorted by descending similarity)
    closest_indices = np.argsort(similarities, axis=1)[:, ::-1][:, :n_results]

    # Prepare the results
    results = {
        "ids": [filtered_ids[i] for i in closest_indices.flatten()],
        "documents": [filtered_documents[i] for i in closest_indices.flatten()],
        "metadata": [filtered_metadata[i] for i in closest_indices.flatten()],
        "distances": [similarities[0, i] for i in closest_indices.flatten()]
    }

    # Include only the specified fields
    filtered_results = {}
    if "embeddings" in include:
        filtered_results["embeddings"] = [filtered_embeddings[i] for i in closest_indices.flatten()]
    if "metadatas" in include:
        filtered_results["metadata"] = [filtered_metadata[i] for i in closest_indices.flatten()]
    if "documents" in include:
        filtered_results["documents"] = [filtered_documents[i] for i in closest_indices.flatten()]
    if "distances" in include:
        filtered_results["distances"] = [similarities[0, i] for i in closest_indices.flatten()]

    return filtered_results


"""
# Streamlit UI
st.title("Famous Five Query App")

# Load data from JSON files
json_files = [
    "famous_five_1.json",
    "famous_five_2.json",
    "famous_five_3.json",
    "famous_five_4.json",
    "famous_five_5.json"
]

vector_dict = load_json_files(json_files)

#st.write(vector_dict)

# Query input
query_text = st.text_input("Enter your query:")

if query_text:
    # Generate embeddings for the query text
    query_embeddings = generate_query_embeddings(query_text)

    # Perform a query on the data (this example checks metadata or document data)
    st.write("Performing query...")
    #where = {'source': '01-five-on-a-treasure-island.md'}  # Example filter, adjust as needed
    results = query_vector_dict(
                                vector_dict, 
                                query_embeddings = query_embeddings,
                                n_results=3 
                                #,where=where
                                )

    # Display the results
    st.write(results)
    
    for i in range(len(results['distances'])):
        st.write(f"Document: {results['documents'][i]}")
        st.write(f"Metadata: {results['metadata'][i]}")
        st.write(f"Distance: {results['distances'][i]}")
        
        st.write("---")
        
else:
    st.write("Please enter a query to get results.")



"""












from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append(Document(page_content=content, metadata={"source": filename}))
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    st.write(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

st.title("Document Embedding Page")
st.markdown("""
This page allows you to upload Markdown documents (.md) and split them into smaller chunks for embedding.
""")

with st.form("document_input"):
    uploaded_files = st.file_uploader(
        "Upload Markdown Files", type=["md"], accept_multiple_files=True
    )

    row_1 = st.columns([2, 1])
    with row_1[0]:
        chunk_size = st.number_input(
            "Chunk Size", value=300, min_value=0, step=1
        )

    with row_1[1]:
        chunk_overlap = st.number_input(
            "Chunk Overlap", value=100, min_value=0, step=1
        )

    row_2 = st.columns([2, 1])
    with row_2[0]:
        source_name = st.text_input(
            "Source Name", "Enter the name of the uploaded file or custom name",
            help="If left empty, the uploaded file's name will be used."
        )

    with row_2[1]:
        add_start_index = st.selectbox(
            "Add Start Index", [True, False], index=0,
            help="Choose whether to add the start index to each chunk."
        )

    save_button = st.form_submit_button("Process Documents")

    if save_button:
        if uploaded_files:
            documents = []
            for file in uploaded_files:
                content = file.read().decode("utf-8")
                metadata_source = source_name if source_name.strip() else file.name
                documents.append(Document(page_content=content, metadata={"source": metadata_source}))
    
            st.write(f"Uploaded {len(documents)} documents.")
    
            chunks = split_text(documents, chunk_size, chunk_overlap, add_start_index)
    
            if chunks:
                st.write("Example chunk:")
                st.write(chunks[0].page_content)
                st.write("Metadata:", chunks[0].metadata)
        else:
            st.warning("Please upload at least one Markdown file.")
