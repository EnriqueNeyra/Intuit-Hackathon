from neo4j import GraphDatabase
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configurations
DATA_FILE = "data.txt"  # Change this to your text file
NEO4J_URI = "bolt://localhost:7687"  # Change if using cloud
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

LLAMA3_API_KEY = "your-api-key"
LLAMA3_API_URL = "https://gravityai-hosted-llama3-endpoint"

# Step 1: Load & Split Text
with open(DATA_FILE, "r", encoding="utf-8") as f:
    raw_text = f.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(raw_text)

# Step 2: Convert Text to Embeddings
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(chunks).tolist()  # Convert to list for Neo4J

# Step 3: Store Embeddings in Neo4J
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def store_embeddings():
    with driver.session() as session:
        session.run("MATCH (n:Chunk) DETACH DELETE n")  # Clear old data
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            session.run(
                "CREATE (c:Chunk {id: $id, text: $text, embedding: $embedding})",
                id=i, text=chunk, embedding=embedding
            )


store_embeddings()


# Step 4: Retrieve Most Relevant Chunks
def retrieve_relevant_chunks(query, top_k=3):
    query_embedding = embed_model.encode([query]).tolist()[0]

    query_string = """
    MATCH (c:Chunk)
    WITH c, 
         gds.similarity.cosine(c.embedding, $query_embedding) AS score
    ORDER BY score DESC
    LIMIT $top_k
    RETURN c.text
    """

    with driver.session() as session:
        results = session.run(query_string, query_embedding=query_embedding, top_k=top_k)
        return [record["c.text"] for record in results]


# Step 5: Function to Call LLama3
def query_llama3(question):
    relevant_chunks = retrieve_relevant_chunks(question)
    context = "\n".join(relevant_chunks)

    payload = {
        "prompt": f"Answer based on the following:\n{context}\n\nQuestion: {question}",
        "temperature": 0.7,
        "max_tokens": 256
    }
    response = requests.post(LLAMA3_API_URL, json=payload, headers={"Authorization": f"Bearer {LLAMA3_API_KEY}"})
    return response.json().get("text", "Error retrieving response")


# Step 6: Run a Query
if __name__ == "__main__":
    question = input("Ask a question: ")
    answer = query_llama3(question)
    print("\nAnswer:", answer)
