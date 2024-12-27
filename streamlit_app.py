import streamlit as st 
import sqlite3
import json
import pickle
import numpy as np
import os

def load_from_sqlite(db_base_path):
    """
    Load data from multiple SQLite databases into a single dictionary.

    Args:
        db_base_path: Base path for the SQLite database files (e.g., '/path/to/db_base').

    Returns:
        A dictionary containing combined data from all databases.
    """
    ids = []
    documents = []
    metadata = []
    embeddings = []

    # Find all database files matching the base path pattern
    db_files = [f for f in os.listdir(os.path.dirname(db_base_path)) if f.startswith(os.path.basename(db_base_path)) and f.endswith(".db")]

    if not db_files:
        print("No database files found.")
        return {
            "ids": ids,
            "documents": documents,
            "metadata": metadata,
            "embeddings": np.array(embeddings)
        }

    # Iterate through the sorted database files
    for db_file in sorted(db_files):
        db_path = os.path.join(os.path.dirname(db_base_path), db_file)

        # Establish a connection to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
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
            print(f"Error reading from {db_path}: {e}")

        finally:
            # Close the database connection
            conn.close()
            print(f"SQLite connection to {db_path} closed.")

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



# Example usage
DB_PATH = "https://github.com/sivagugan30/Ask-RAG-LLM/blob/main/sqlite/famous_five"
vector_dict3 = load_from_sqlite(DB_PATH)




