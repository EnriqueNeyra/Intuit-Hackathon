from neo4j import GraphDatabase
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import HostedAPI

# Configurations
DATA_FILE = "solar_system.txt"  # Change this to your text file
NEO4J_URI = "neo4j+ssc://ce8a4d76.databases.neo4j.io"  # Change if using cloud
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "G8_ygSDrSoTDebg3iFKiUpMtbcd_hixaA335eGdwzMI"
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

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
    # payload = {
    #     "prompt": f"Answer based on the following:\n{context}\n\nQuestion: {question}",
    #     "temperature": 0.7,
    #     "max_tokens": 256
    # }

    #response = requests.post(LLAMA3_API_URL, json=payload, headers={"Authorization": f"Bearer {LLAMA3_API_KEY}"})

    prompt = f"Answer based on the following:\n{context}\n\nQuestion: {question}"
    file = open("prompt.txt", "w")
    file.write(prompt)
    file.close()

    response = HostedAPI.postJob('prompt.txt')
    return response.json().get("text", "Error retrieving response")


# Step 6: Run a Query
if __name__ == "__main__":
    # Step 1: Load & Split Text
    store_embeddings()
    question = input("Ask a question: ")
    answer = query_llama3(question)
    print("\nAnswer:", answer)
