import streamlit as st
import openai
import chromadb
#import sqlite3
import json
import pickle
from openai import OpenAI
#from langchain.schema import Document
#from langchain.text_splitter import RecursiveCharacterTextSplitter


# Manually set the API key for testing purposes
os.environ["OPENAI_API_KEY"] = st.secrets["KEY"]

# Load ChromaDB Collection from SQLite DB
def create_chromadb_from_sqlite(db_path2, table_name2):
    # Initialize the ChromaDB client
    client2 = chromadb.Client()

    # Delete and create new collection
    client2.delete_collection(name="famous-five")
    collection2 = client2.create_collection(name="famous-five")

    # Connect to the SQLite database
    conn2 = sqlite3.connect(db_path2)
    cursor2 = conn2.cursor()

    # Query the table for data (id, document, metadata, embedding)
    query2 = f"SELECT id, document, metadata, embedding FROM {table_name2}"
    cursor2.execute(query2)
    rows2 = cursor2.fetchall()

    # Close the database connection
    conn2.close()

    # Extract data into separate lists
    ids2 = []
    documents2 = []
    metadatas2 = []
    embeddings2 = []

    for row2 in rows2:
        ids2.append(row2[0])  # Assuming 'id' is the first column
        documents2.append(row2[1])  # Assuming 'document' is the second column
        metadatas2.append(eval(row2[2]) if row2[2] else {})  # Convert metadata string to dictionary

        # Convert the embedding from BLOB (assuming pickled format in DB)
        embedding2 = pickle.loads(row2[3]).tolist() if row2[3] else None
        embeddings2.append(embedding2)

    # Add data to collection2
    collection2.add(
        ids=ids2,
        documents=documents2,
        metadatas=metadatas2,
        embeddings=embeddings2  # Add embeddings as well
    )

    print(f"Collection 'famous-five' created with {len(ids2)} records in collection2.")
    
    return ids2, documents2, metadatas2, embeddings2, collection2

# Load collection and setup environment
db_path = "https://github.com/sivagugan30/Ask-RAG-LLM/sqlite/collection.db"  # SQLite path
table_name = "collection"

# Initialize collection
ids2, documents2, metadatas2, embeddings2, collection2 = create_chromadb_from_sqlite(db_path, table_name)

# Streamlit app
def main():
    st.title("Explore Famous Five Adventures")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Home", "Instructions", "Relevance Dashboard", "Character Analysis"])

    # Home Tab
    if selection == "Home":
        st.header("Welcome to the Famous Five Adventure!")
        st.write("""
        Explore the world of Famous Five with Enid Blyton's first four novels. Ask questions about the characters, mysteries, and adventures, and we'll find the most relevant sections from the books to answer your queries.
        """)
        
        # User query input
        query = st.text_input("Ask a question about Famous Five novels:", "")
        if query:
            query_embeddings = OpenAI().embeddings.create(
                input=query,
                model="text-embedding-3-small"  # Specify the embedding model
            ).data[0].embedding

            results = collection2.query(
                query_embeddings=query_embeddings,
                n_results=3
            )
            
            if results['documents']:
                st.write("### Top 3 results:")
                for i, doc in enumerate(results['documents']):
                    st.write(f"**Result {i+1}:**")
                    st.write(doc)
                    st.write(f"**Source:** {results['metadatas'][i][0]['source']}")
                    st.write(f"**Start Index:** {results['metadatas'][i][0]['start_index']}")
            else:
                st.write("Sorry, no relevant results found.")

    # Instructions Tab
    elif selection == "Instructions":
        st.header("Instructions")
        st.write("""
        This app allows you to ask questions related to the Famous Five novels (Books 1-4). 
        Enter a query in the Home tab and the system will return the most relevant excerpts from the books based on your question.
        Use the Relevance Dashboard to see how relevant the documents are and the Character Analysis tab to learn about the main characters.
        """)
        
    # Relevance Dashboard Tab
    elif selection == "Relevance Dashboard":
        st.header("Relevance Dashboard")
        
        # Display relevance scores as a heatmap or bar chart
        if query:
            query_embeddings = OpenAI().embeddings.create(
                input=query,
                model="text-embedding-3-small"
            ).data[0].embedding

            results = collection2.query(
                query_embeddings=query_embeddings,
                n_results=3
            )
            
            if results['documents']:
                st.write("### Relevance Scores:")
                for i, doc in enumerate(results['documents']):
                    score = results['metadatas'][i][0]['start_index']  # Use this or a custom score for relevance
                    st.write(f"**Result {i+1}:** {score}")
            else:
                st.write("No results to display.")

    # Character Analysis Tab
    elif selection == "Character Analysis":
        st.header("Character Analysis")
        
        # Allow the user to choose a character (e.g., Julian, George, Anne)
        character = st.selectbox("Choose a character", ["Julian", "George", "Anne", "Dick", "Timmy"])
        
        # Based on character, display relevant information from the collection
        st.write(f"Displaying results for {character}.")
        if character:
            character_query = f"Tell me about {character}'s role in the Famous Five series."
            
            # Generate query embedding
            character_query_embedding = OpenAI().embeddings.create(
                input=character_query,
                model="text-embedding-3-small"
            ).data[0].embedding
            
            results = collection2.query(
                query_embeddings=character_query_embedding,
                n_results=3
            )
            
            if results['documents']:
                st.write("### Character Insights:")
                for i, doc in enumerate(results['documents']):
                    st.write(f"**Result {i+1}:** {doc}")
                    st.write(f"**Source:** {results['metadatas'][i][0]['source']}")
                    st.write(f"**Start Index:** {results['metadatas'][i][0]['start_index']}")
            else:
                st.write(f"No relevant information found for {character}.")

if __name__ == "__main__":
    main()
