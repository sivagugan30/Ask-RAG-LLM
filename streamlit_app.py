import requests
import sqlite3
import json
import pickle
import numpy as np
import io
import streamlit as st

# List of database URLs
db_urls = [
    "https://github.com/sivagugan30/Ask-RAG-LLM/raw/main/sqlite/famous_five_1.db"
]

ids = []
documents = []
metadata = []
embeddings = []

for db_url in db_urls:
    st.write(f"Fetching database from: {db_url}")
    try:
        # Fetch database content
        response = requests.get(db_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        st.write(f"Successfully fetched database from {db_url}")
        db_file = io.BytesIO(response.content)  # Load content into a file-like object
    except requests.exceptions.RequestException as e:
        st.write(f"Failed to fetch database from {db_url}: {e}")
        continue  # Skip this URL if fetching fails

    # Establish a connection to the SQLite database in memory
    st.write("Connecting to SQLite database in memory.")
    conn = sqlite3.connect(":memory:")  # In-memory SQLite database
    cursor = conn.cursor()

    try:
        # Load database content into SQLite memory
        st.write("Loading database into memory.")
        conn.executescript(db_file.getvalue().decode())

        # Query the data
        st.write("Querying the data from the table 'collection'.")
        cursor.execute("SELECT id, document, metadata, embedding FROM collection")
        rows = cursor.fetchall()
        st.write(f"Fetched {len(rows)} rows from the database.")

        # Process rows
        for row in rows:
            ids.append(row[0])
            documents.append(row[1])
            metadata.append(json.loads(row[2]))  # Deserialize JSON metadata
            embeddings.append(pickle.loads(row[3]))  # Deserialize BLOB embedding

        st.write(f"Processed {len(rows)} rows successfully.")
    except Exception as e:
        st.write(f"Error reading from the database at {db_url}: {e}")
    finally:
        # Close the database connection
        conn.close()
        st.write(f"SQLite connection to {db_url} closed.")

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
