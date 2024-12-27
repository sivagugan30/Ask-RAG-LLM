import requests
import json
import numpy as np
import io
import streamlit as st

# List of JSON URLs
json_urls = [
    "famous_five_1.json"
]

ids = []
documents = []
metadata = []
embeddings = []

for json_url in json_urls:
    st.write(f"Fetching JSON data from: {json_url}")
    try:
        # Fetch JSON content
        response = requests.get(json_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        st.write(f"Successfully fetched JSON data from {json_url}")
        json_data = response.json()  # Parse the JSON content
    except requests.exceptions.RequestException as e:
        st.write(f"Failed to fetch JSON data from {json_url}: {e}")
        continue  # Skip this URL if fetching fails

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
st.write(vector_dict)
