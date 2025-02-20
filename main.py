from neo4j import GraphDatabase
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import HostedAPI
import json

# Configurations
DATA_FILE = "solar_system.txt"  # Change this to your text file
NEO4J_URI = "neo4j+ssc://ce8a4d76.databases.neo4j.io"  # Change if using cloud
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "G8_ygSDrSoTDebg3iFKiUpMtbcd_hixaA335eGdwzMI"
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
API_URL = "https://on-demand.gravity-ai.com/"
API_CREATE_JOB_URL = API_URL + 'api/v1/jobs'
API_GET_JOB_RESULT_URL = API_URL + 'api/v1/jobs/result-link'
API_KEY = "GAI-wmBiBQ7e.bm3OFeWHioeHUprWJ30ij218sF-GAc"  # Use API Keys tab to generate

config = {
    "version": "0.0.1",
    # Optional - if omitted, the latest version will be used; Set to a specific version number (i.e. 0.0.1, 0.0.2, 1.0.1, etc. Check versions on Hosted Inference page Version dropdown)
    "mimeType": "application/json; header=present",  # 'text/plain', etc. Change based on your file type
}

requestHeaders = {
    'x-api-key': API_KEY
}

API_GET_JOB_RESULT_URL = API_URL + 'api/v1/jobs/result-link'

# ------------------------   RAG Implementation -----------------

def store_embeddings():
    # Load & Split Text
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(raw_text)
    embeddings = embed_model.encode(chunks).tolist()  # Convert to list for Neo4J

    with driver.session() as session:
        session.run("MATCH (n:Chunk) DETACH DELETE n")  # Clear old data
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            session.run(
                "CREATE (c:Chunk {id: $id, text: $text, embedding: $embedding})",
                id=i, text=chunk, embedding=embedding
            )


# Step 4: Retrieve Most Relevant Chunks
def retrieve_relevant_chunks(query, top_k=3):
    query_embedding = embed_model.encode([query]).tolist()[0]
    query_string = """
    MATCH (c:Chunk)
    WITH c, $query_embedding AS query_embedding, c.embedding AS embedding
    // Compute the dot product between the two vectors
    WITH c, query_embedding, embedding,
         reduce(dot = 0.0, i IN range(0, size(embedding)) | dot + embedding[i] * query_embedding[i]) AS dot,
         // Compute the norm of the stored embedding
         sqrt(reduce(sum = 0.0, i IN embedding | sum + i * i)) AS norm_embedding,
         // Compute the norm of the query embedding
         sqrt(reduce(sum = 0.0, i IN query_embedding | sum + i * i)) AS norm_query
    WITH c, dot/(norm_embedding * norm_query) AS score
    ORDER BY score DESC
    LIMIT $top_k
    RETURN c.text AS chunk_text
    """


    with driver.session() as session:
        results = session.run(query_string, query_embedding=query_embedding, top_k=top_k)
        return [record["chunk_text"] for record in results]


# Step 5: Function to Call LLama3
def query_llama3(question):
    relevant_chunks = retrieve_relevant_chunks(question)
    context = "\n".join(relevant_chunks)

    # TODO turn prompt into txt file
    payload = {
        "dialog": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": f"Answer based on the following:\n{context}\n\nQuestion: {question}"
            }
            ],
            "max_gen_len": None,
            "temperature": 0.7,
            "top_p": 0.9
    }

    with open("prompt.json", "w") as json_file:
        json.dump(payload, json_file, indent=4)

    response = HostedAPI.postJob('prompt.json')

    return HostedAPI.downloadResult(response)


# Step 6: Run a Query
if __name__ == "__main__":
    # Step 1: Load & Split Text
    store_embeddings()
    question = input("Ask a question: ")
    answer = query_llama3(question)
    print("\nAnswer:", answer)
