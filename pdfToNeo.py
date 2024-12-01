import os
import logging
from dotenv import load_dotenv
from pdfminer.high_level import extract_text
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from py2neo import Graph
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import List, Optional
from neo4j import GraphDatabase, PoolConfig
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Node(BaseModel):
    id: str
    type: str

class Relationship(BaseModel):
    source: Node
    target: Node
    type: str

class GraphDocument(BaseModel):
    nodes: List[Node]
    relationships: List[Relationship]

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    try:
        return extract_text(file_path)
    except Exception as e:
        logger.error(f"An error occurred while extracting text: {e}")
        return ""

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,  # Increase chunk size for fewer API calls
        chunk_overlap=300,  # Increase overlap for better context
        length_function=len
    )
    return text_splitter.split_text(text)

def sanitize_relationship_type(relationship_type):
    # Replace hyphens with underscores or another acceptable character
    return relationship_type.replace('-', '_')

def escape_string(value: str) -> str:
    """Escape apostrophes in string values."""
    return value.replace("'", "\\'")

def generate_cypher_queries(graph_document: GraphDocument) -> List[str]:
    """Generate Cypher queries to create nodes and relationships from graph document."""
    queries = []
    for node in graph_document.nodes:
        node_id = escape_string(node.id)
        node_type = escape_string(node.type)
        queries.append(f'MERGE (n:Node {{id: "{node_id}", type: "{node_type}"}})')

    for relationship in graph_document.relationships:
        source_id = escape_string(relationship.source.id)
        target_id = escape_string(relationship.target.id)
        rel_type = sanitize_relationship_type(relationship.type)
        queries.append(
            f'MATCH (a:Node {{id: "{source_id}"}}), (b:Node {{id: "{target_id}"}}) '
            f'MERGE (a)-[r:{rel_type}]->(b)'
        )
    return queries


def run_queries_in_batches(queries, batch_size=100):
    total_queries = len(queries)
    for i in range(0, total_queries, batch_size):
        batch = queries[i:i + batch_size]
        with driver.session() as session:
            with session.begin_transaction() as tx:
                for query in batch:
                    tx.run(query)
                tx.commit()
        logger.info(f"Executed batch {i // batch_size + 1}/{(total_queries // batch_size) + 1}")

def process_chunk(chunk, llm_transformer):
    retry_count = 3
    for attempt in range(retry_count):
        try:
            documents = [Document(page_content=chunk)]
            graph_documents = llm_transformer.convert_to_graph_documents(documents)

            # Log raw response for debugging
            for doc in graph_documents:
                logger.info(f"Raw graph document on attempt {attempt + 1}: {doc}")

            return graph_documents
        except Exception as e:
            logger.error(f"Error processing chunk on attempt {attempt + 1}: {e}")
            logger.error(f"Chunk content: {chunk[:500]}")  # Log first 500 characters of the chunk
    return []

def save_embeddings_to_neo4j(vectorstore, text_chunks):
    # Serialize FAISS index to a byte array
    index_bytes = faiss.serialize_index(vectorstore.index)

    # Convert the byte array to a hex string
    index_str = index_bytes.tobytes().hex()

    with driver.session() as session:
        # Update the node with id 0 with embeddings and text chunks
        session.run(
            """
            MATCH (n) WHERE id(n) = 0
            SET n.embeddings = $embeddings, n.text_chunks = $text_chunks
            """,
            embeddings=index_str,
            text_chunks=text_chunks
        )

def main():
    # Load environment variables from .env file
    dotenv_path = os.path.join("/home/fafnir/Alpha/_Python/Python Current/Monalisa", '.env')
    load_dotenv(dotenv_path)

    # Neo4j database connection with custom pool configuration
    global driver
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"), 
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
        max_connection_pool_size=50,  # Increase pool size
        connection_acquisition_timeout=120  # Increase timeout
    )

    # Set up the Language Model and Graph Transformer
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
    llm_transformer = LLMGraphTransformer(llm=llm)

    # Input PDF file name
    file_name = input("Enter the name of the PDF file (including .pdf extension): ")
    file_path = os.path.join(r"/home/fafnir/Alpha/_Python/Python Current/Monalisa", file_name)
    
    # Check if file exists
    if not os.path.exists(file_path):
        print("File does not exist. Please check the file name and path.")
        return
    
    # Extract text from the PDF
    extracted_text = extract_text_from_pdf(file_path)
    if not extracted_text:
        print("Failed to extract text from the PDF.")
        return

    # Split text into chunks
    text_chunks = get_text_chunks(extracted_text)

    # Process chunks and collect all graph documents
    all_graph_documents = []
    for chunk in text_chunks:
        chunk_graph_documents = process_chunk(chunk, llm_transformer)
        all_graph_documents.extend(chunk_graph_documents)

    if not all_graph_documents:
        print("No valid graph documents generated.")
        return

    # Output the nodes and relationships for the first document as a sample
    if all_graph_documents:
        print(f"Nodes: {all_graph_documents[0].nodes}")
        print(f"Relationships: {all_graph_documents[0].relationships}")

    # Generate and execute Cypher queries
    all_queries = []
    for graph_document in all_graph_documents:
        all_queries.extend(generate_cypher_queries(graph_document))

    run_queries_in_batches(all_queries)

    print("Graph documents successfully added to the database.")
    
    # Process and upload embeddings
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    save_embeddings_to_neo4j(vectorstore, text_chunks)

    print("Embeddings uploaded successfully!")

if __name__ == "__main__":
    main()
