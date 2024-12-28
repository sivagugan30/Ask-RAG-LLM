import json
import numpy as np
import streamlit as st
from openai import OpenAI
import os 

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
        st.write(f"Loading JSON data from: {json_file}")
        try:
            # Open and read the JSON file
            with open(json_file, 'r') as file:
                json_data = json.load(file)  # Parse the JSON content
            st.write(f"Successfully loaded JSON data from {json_file}")
        except Exception as e:
            st.write(f"Failed to load JSON data from {json_file}: {e}")
            continue  # Skip this file if reading fails

        # Process the data from the JSON
        try:
            st.write("Processing JSON data.")
            ids.extend(json_data["ids"])  # Append data to the existing list
            documents.extend(json_data["documents"])
            metadata.extend(json_data["metadata"])
            embeddings.extend(json_data["embeddings"])
            st.write(f"Processed data with {len(json_data['ids'])} rows successfully.")
        except KeyError as e:
            st.write(f"Error: Key {e} not found in the JSON data.")
        except Exception as e:
            st.write(f"Error processing the JSON data: {e}")

    # Convert embeddings to a NumPy array
    st.write("Converting embeddings to a NumPy array.")
    embeddings_array = np.array(embeddings)

    # Combine all data into a dictionary
    st.write("Combining all data into a single dictionary.")
    vector_dict = {
        "ids": ids,
        "documents": documents,
        "metadata": metadata,
        "embeddings": embeddings_array
    }

    st.write("Data loading complete.")
    st.write(f"Shape of embeddings: {vector_dict['embeddings'].shape}")
    st.write(f"Number of IDs: {len(vector_dict['ids'])}")

    return vector_dict


# Function to query the vector_dict with a 'where' filter
def query_vector_dict(vector_dict, where=None):
    ids = vector_dict["ids"]
    documents = vector_dict["documents"]
    metadata = vector_dict["metadata"]
    embeddings = vector_dict["embeddings"]

    # Filter metadata based on the 'where' condition
    if where:
        matching_indices = []
        for i, metadata_item in enumerate(metadata):
            if all(metadata_item.get(key) == value for key, value in where.items()):
                matching_indices.append(i)

        # Retrieve the documents and embeddings for the matching indices
        matching_documents = [documents[i] for i in matching_indices]
        matching_metadata = [metadata[i] for i in matching_indices]
        matching_embeddings = embeddings[matching_indices]

        results = {
            "ids": [ids[i] for i in matching_indices],
            "documents": matching_documents,
            "metadata": matching_metadata,
            "embeddings": matching_embeddings
        }

        return results
    else:
        return {
            "ids": ids,
            "documents": documents,
            "metadata": metadata,
            "embeddings": embeddings
        }


# Function to generate embeddings for a query using OpenAI API
def generate_query_embeddings(query_text):
    query_embeddings = OpenAI().embeddings.create(
        input=query_text,
        model="text-embedding-3-small"  # Specify the embedding model
    ).data[0].embedding

    query_embeddings = np.array(query_embeddings).reshape(1, -1)
    return query_embeddings



st.title("Famous Five Query App")

# Load data from JSON files
vector_dict = load_json_files(json_files)

# Query input
query_text = st.text_input("Enter your query:", "how many children are at breakfast-table")

if query_text:
    # Generate embeddings for the query text
    query_embeddings = generate_query_embeddings(query_text)

    # Perform a query on the data (this example checks metadata or document data)
    st.write("Performing query...")
    where = {'source': '01-five-on-a-treasure-island.md'}  # Example filter, adjust as needed
    results = query_vector_dict(vector_dict, where)

    # Display the results
    st.write(f"Found {len(results['ids'])} matching results:")
    for i in range(len(results['ids'])):
        st.write(f"ID: {results['ids'][i]}")
        st.write(f"Document: {results['documents'][i]}")
        st.write(f"Metadata: {results['metadata'][i]}")
        st.write("---")
        
else:
    st.write("Please enter a query to get results.")
