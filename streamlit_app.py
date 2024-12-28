import json
import numpy as np
import streamlit as st

# List of JSON file paths (local directory)
json_files = [
    "famous_five_1.json",
    "famous_five_2.json",
    "famous_five_3.json",
    "famous_five_4.json",
    "famous_five_5.json"
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
vector_dict1 = {
    "ids": ids,
    "documents": documents,
    "metadata": metadata,
    "embeddings": embeddings_array
}

st.write("Data loading complete. Here's the combined dictionary:")
st.write(f"Shape of embeddings: {vector_dict1['embeddings'].shape}")
st.write(f"Number of IDs: {len(vector_dict1['ids'])}")


st.write("")
st.write(vector_dict1)


