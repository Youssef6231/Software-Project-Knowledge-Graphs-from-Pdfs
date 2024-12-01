from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv("/home/fafnir/Alpha/_Python/Python Current/Monalisa/.env")

# Get connection details from .env file
uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

# Create a driver instance
driver = GraphDatabase.driver(uri, auth=(user, password))

# Define a function to test the connection
def test_connection():
    try:
        with driver.session() as session:
            result = session.run("RETURN 'Connection successful' AS message")
            for record in result:
                print(record["message"])
    except Exception as e:
        print("Failed to connect to Neo4j:", e)
    finally:
        driver.close()

# Run the test
test_connection()
