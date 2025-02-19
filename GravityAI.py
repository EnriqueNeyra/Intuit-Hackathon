from neo4j import GraphDatabase
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configurations
DATA_FILE = "data.txt"  # Change this to your text file
NEO4J_URI = "neo4j+s://ce8a4d76.databases.neo4j.io"  # Change if using cloud
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "G8_ygSDrSoTDebg3iFKiUpMtbcd_hixaA335eGdwzMI"

# ------------------------   LLama API Access -----------------
import requests
import sys
import json

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


def postJob(inputFilePath):
    # Post a new job (file) to the api
    inputFile = open(inputFilePath, 'rb')
    files = {
        "file": inputFile,
    }

    data = {
        'data': json.dumps(config)
    }

    print("Creating job...")
    r = requests.request("POST", API_CREATE_JOB_URL, headers=requestHeaders, data=data, files=files)
    print(r.status_code)
    result = r.json()
    if (result.get('isError', False)):
        print("Error: " + result.get('errorMessage'))
    #     raise Exception("Error: " + result.errorMessage)
    if (result.get('data').get('statusMessage') != "success"):
        print("Job Failed: " + result.get('data').get('errorMessage'))
        raise Exception("Job Failed: " + result.get('data').get('errorMessage'))
    return result.get('data').get('id')


def downloadResult(jobId, outFilePath):
    url = API_GET_JOB_RESULT_URL + "/" + jobId
    r = requests.request("GET", url, headers=requestHeaders)
    link = r.json()
    if (link.get('isError')):
        print("Error: " + link.get('errorMessage'))
        raise Exception("Error: " + link.get('errorMessage'))

    result = requests.request("GET", link.get('data'))
    open(outFilePath, 'wb').write(result.content)


jobId = postJob(sys.argv[1])
downloadResult(jobId, sys.argv[2])

# ------------------------   RAG Implementation -----------------

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
