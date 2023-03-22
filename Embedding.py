import json
import openai
import os
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Function to retrieve Ada-002 embeddings
def get_embeddings(query, model="text-embedding-ada-002"):
    response = openai.Embedding.create(
        model=model,
        input=query
    )
    
    return response['data'][0]['embedding']

# Function to generate a response using GPT-3.5 Turbo and embeddings
def generate_response(context_messages, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(model=model,
        max_tokens=1000,
        temperature=1.2,
        messages=context_messages,
    )
    
    return response.choices[0]['message']['content'].strip()

def embbeding_exists(conn, string):
    c = conn.cursor()
    query = f"SELECT COUNT(*) FROM memories WHERE text = ?"
    c.execute(query, (string,))
    count = c.fetchone()[0]
    return count > 0

# Function to store embeddings in SQLite database
def store_embeddings(embeddings, text, conn):
    cursor = conn.cursor()
    
    embeddings = json.dumps(embeddings)
    
    cursor.execute("INSERT INTO memories (embedding, text) VALUES (?, ?)", (embeddings, text))
    conn.commit()

# Function to retrieve matching memory from SQLite database
def retrieve_matching_memory(embeddings, conn, threshold=0.8, limit=8):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM memories")
    rows = cursor.fetchall()

    similarities = []
    for row in rows:
        stored_embeddings = np.array(json.loads(row[1])).reshape(1, -1)
        embeddings_array = np.array(embeddings)
        similarity = cosine_similarity(embeddings_array.reshape(1, -1), stored_embeddings)[0, 0]
        if similarity >= threshold:
            similarities.append((similarity, row[2]))

    if similarities:
        similarities = sorted(similarities, key=lambda x: x[0], reverse=True)[:limit]
        return '\n'.join([f'{sim[1]} ({sim[0]:.2f})' for sim in similarities])
    else:
        return None

# Initialize SQLite database
def initialize_database(database_name="memories.db"):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY AUTOINCREMENT, embedding TEXT UNIQUE, text TEXT)")
    conn.commit()
    return conn

# Load SQLite database into memory
def load_database_to_memory(database_name="memories.db"):
    # Connect to the on-disk database
    disk_conn = sqlite3.connect(database_name)

    # Connect to the in-memory database
    mem_conn = sqlite3.connect(':memory:')

    # Copy data from the on-disk database to the in-memory database
    query = "".join(line for line in disk_conn.iterdump())
    mem_conn.executescript(query)

    # Close the connection to the on-disk database
    disk_conn.close()

    return mem_conn
    
def save_memory_database_to_disk(mem_conn, database_name="memories.db"):
    # Connect to the on-disk database
    disk_conn = sqlite3.connect(database_name)

    # Check if the memories table exists in the on-disk database
    cursor = disk_conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories';")
    result = cursor.fetchone()

    # If the memories table exists, drop it
    if result:
        cursor.execute("DROP TABLE memories;")
        print("Dropped existing 'memories' table in on-disk database.")

    # Copy data from the in-memory database to the on-disk database
    query = "".join(line for line in mem_conn.iterdump())
    disk_conn.executescript(query)
    print("Copied data from in-memory database to on-disk database.")

    # Commit changes and close the connection to the on-disk database
    disk_conn.commit()
    disk_conn.close()


# Example usage
conn = initialize_database()
conn = load_database_to_memory()

# Don't forget to set your API key!
openai.api_key = "INSERT API KEY"

# Storing a memory (Example based on knowledge from after 2021)
memory_text = "Dragons of Stormwreck Isle is a 5th edition introductory adventure included in the Dungeons & Dragons Starter Set (2022) boxed set. It is designed for 1–5 player-characters, taking them from 1st level to 3rd as they explore the eponymous Stormwreck Isle, a new location within the Forgotten Realms. To a greater degree than previous adventures intended as entry points for 5th edition Dungeons & Dragons, such as those included in the original Dungeons & Dragons Starter Set and in the Dungeons & Dragons Essentials Kit, Dragons of Stormwreck Isle is intended primarily to teach how to play the game of Dungeons & Dragons. This means that the adventure booklet contains many more tips and notes for the DM about how to run certain scenes or encounters, how to treat players, and how to engage players. The adventure is intended to be played with the pre-generated characters included in the boxed set, and includes motivations and hooks specific to backgrounds included on those character sheets."

# If the memory doesn't exist, overwrite it
if embbeding_exists(conn, memory_text) == False:
    print('Requesting embedding of new memory.')
    memory_embeddings = get_embeddings(memory_text)
    store_embeddings(memory_embeddings, memory_text, conn)

    # Save the in-memory database back to the on-disk database
    save_memory_database_to_disk(conn)

# Retrieving a matching memory
user_input = "What do you know about Dragons of Stormwreck Isle?"
input_embeddings = get_embeddings(user_input)
matched_memory = retrieve_matching_memory(input_embeddings, conn)

# Generating a response using the matched memory
if matched_memory:
    context_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "assistant", "content": matched_memory},
        {"role": "user", "content": user_input}
    ]
    response = generate_response(context_messages)
    print(response)
