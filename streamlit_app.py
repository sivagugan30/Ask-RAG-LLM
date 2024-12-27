import json
import numpy as np
import streamlit as st

# List of JSON file paths (local directory)
json_files = [
    "famous_five_1.json",
    "famous_five_2.json",
    "famous_five_3.json",
    "famous_five_4.json",
    "famous_five_5.json"# Update this path as necessary
]

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
        
        ids = json_data["ids"]
        documents = json_data["documents"]
        metadata = json_data["metadata"]
        embeddings = json_data["embeddings"]
        
        st.write(f"Processed data with {len(ids)} rows successfully.")
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

st.write("Data loading complete. Here's the combined dictionary:")
st.write(vector_dict['embeddings'])
