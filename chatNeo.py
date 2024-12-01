import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import faiss
import streamlit as st
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.docstore.document import Document
import tempfile

# Load environment variables from .env file
load_dotenv("/home/fafnir/Alpha/_Python/Python Current/Monalisa/.env")

# Neo4j database connection
driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")))

def extract_entity(question, entity_type):
    # Simple entity extraction using keyword search (can be replaced with a more advanced method)
    for etype in entity_type:
        if etype.lower() in question.lower():
            return etype
    return None

def get_answer_from_neo4j(question):
    with driver.session() as session:
        if "release" in question or "released" in question:
            result = session.run("MATCH (n) WHERE n.title =~ '(?i).*release.*' RETURN n")
            for record in result:
                return record['n']['title']
        else:
            return None

def load_vectorstore_from_neo4j():
    with driver.session() as session:
        result = session.run("MATCH (n) WHERE id(n) = 0 RETURN n.embeddings AS embeddings, n.text_chunks AS text_chunks")
        record = result.single()
        if record is None:
            st.error("No embeddings found in Neo4j.")
            return None

        index_str = record['embeddings']
        texts = record['text_chunks']

        index_bytes = bytes.fromhex(index_str)
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(index_bytes)
            temp_file_path = temp_file.name

        index = faiss.read_index(temp_file_path)
        os.remove(temp_file_path)  # Clean up the temporary file

        embeddings = OpenAIEmbeddings()
        documents = [Document(page_content=text) for text in texts]
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
        index_to_docstore_id = {i: str(i) for i in range(len(documents))}

        vectorstore = FAISS(embeddings, index, docstore, index_to_docstore_id)
        return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    answer = get_answer_from_neo4j(user_question)
    if not answer:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(f"**User**: {message.content}")
            else:
                st.write(f"**Bot**: {message.content}")
    else:
        st.write(f"**Bot**: {answer}")

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Neo4j", page_icon=":books:")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Neo4j :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    if st.session_state.conversation is None:
        with st.spinner("Loading embeddings from Neo4j"):
            vectorstore = load_vectorstore_from_neo4j()
            if vectorstore:
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Embeddings loaded and processed successfully!")

if __name__ == '__main__':
    main()
