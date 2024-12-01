Python running commands:


python pdfToNeo.py

streamlit run chatNeo.py

streamlit run chatNeo4Eval.py



After running pdfToNeo.py, you can then paste the pdf name in the terminal:

2405.00330v1.pdf
Jacob's Law .pdf



Neo4j commands:

MATCH (n)
OPTIONAL MATCH (n)-[r]-()
RETURN n, r


Delete all:

MATCH (n)
DETACH DELETE n