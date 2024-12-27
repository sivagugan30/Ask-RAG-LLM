import requests
import sqlite3
import json
import pickle
import numpy as np
import io
import streamlit as st

def fetch_db_from_github(url):
    """
    Fetch the SQLite database file from a GitHub raw URL.

    Args:
        url: The GitHub raw URL of the SQLite database file.

    Returns:
        A BytesIO object containing the database file content if successful, else None.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise error for failed requests
        print(f"Fetched database from {url}")
        return io.BytesIO(response.content)  # Return the content as a file-like object
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch database from {url}: {e}")
        return None

def load_from_sqlite_list(db_urls):
    """
    Load data from a list of SQLite database URLs into a single dictionary.

    Args:
        db_urls: List of GitHub raw URLs pointing to SQLite database files.

    Returns:
        A dictionary containing combined data from all databases.
    """
    ids = []
    documents = []
    metadata = []
    embeddings = []

    for db_url in db_urls:
        # Fetch the database file content
        db_file = fetch_db_from_github(db_url)
        if not db_file:
            print(f"Skipping database at {db_url}.")
            continue  # Skip this URL if fetching fails

        # Establish a connection to the SQLite database in memory
        conn = sqlite3.connect(":memory:")  # Use in-memory database
        cursor = conn.cursor()

        try:
            # Load database content into SQLite memory
            with open(db_url, "rb") as f: 
                conn.executescript(f.read().decode())

            # Query the table for data
            cursor.execute("SELECT id, document, metadata, embedding FROM collection")
            rows = cursor.fetchall()

            # Extract data into lists
            for row in rows:
                ids.append(row[0])
                documents.append(row[1])
                metadata.append(json.loads(row[2]))  # Deserialize metadata from JSON
                embeddings.append(pickle.loads(row[3]))  # Deserialize embedding from BLOB

        except Exception as e:
            print(f"Error reading from {db_url}: {e}")

        finally:
            # Close the database connection
            conn.close()
            print(f"SQLite connection to {db_url} closed.")

    # Convert embeddings to a numpy array
    embeddings_array = np.array(embeddings)

    # Combine all into a single dictionary
    vector_dict = {
        "ids": ids,
        "documents": documents,
        "metadata": metadata,
        "embeddings": embeddings_array
    }

    print("Data loaded from all provided SQLite databases into a dictionary.")
    return vector_dict

# Example usage in Streamlit
db_urls = [
    "https://github.com/sivagugan30/Ask-RAG-LLM/raw/main/sqlite/famous_five_1.db",
    "https://github.com/sivagugan30/Ask-RAG-LLM/raw/main/sqlite/famous_five_2.db",
    "https://github.com/sivagugan30/Ask-RAG-LLM/raw/main/sqlite/famous_five_3.db",
]

vector_dict3 = load_from_sqlite_list(db_urls)

st.write("Loaded Data:")
st.write(vector_dict3)
