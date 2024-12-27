import streamlit as st
import requests
import sqlite3
import json
import pickle
import numpy as np
import io


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

def load_from_sqlite_dynamic_urls(base_url):
    """
    Load data from multiple SQLite databases dynamically by checking for sequentially numbered files.

    Args:
        base_url: Base URL for the SQLite database files (e.g., 'https://github.com/<user>/<repo>/raw/main/<path>/famous_five').

    Returns:
        A dictionary containing combined data from all databases.
    """
    ids = []
    documents = []
    metadata = []
    embeddings = []
    
    db_index = 1
    while True:
        # Generate the current database URL
        db_url = f"{base_url}_{db_index}.db"
        
        # Fetch the database file content
        db_file = fetch_db_from_github(db_url)
        if not db_file:
            print(f"No more databases found after {db_url}.")
            break  # Exit the loop if the file does not exist

        # Establish a connection to the SQLite database in memory
        conn = sqlite3.connect(":memory:")  # Use in-memory database
        cursor = conn.cursor()

        try:
            # Load database content into SQLite memory
            conn.executescript(db_file.read().decode("utf-8"))

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

        db_index += 1

    # Convert embeddings to a numpy array
    embeddings_array = np.array(embeddings)

    # Combine all into a single dictionary
    vector_dict = {
        "ids": ids,
        "documents": documents,
        "metadata": metadata,
        "embeddings": embeddings_array
    }

    print("Data loaded from all SQLite databases into a dictionary.")
    return vector_dict

# Example usage in Streamlit
db_base_url = "https://github.com/sivagugan30/Ask-RAG-LLM/raw/main/sqlite/famous_five"
vector_dict3 = load_from_sqlite_dynamic_urls(db_base_url)

st.write("Loaded Data:")
st.write(vector_dict3)
