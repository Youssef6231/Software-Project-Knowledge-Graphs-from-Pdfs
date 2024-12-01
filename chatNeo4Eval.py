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
import json
from convergenceEval import ConvergenceEvaluator

# Load environment variables from .env file
load_dotenv()

# Neo4j database connection
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

# Store conversation logs
conversation_log = {"queries": [], "responses": []}

def load_vectorstore_from_neo4j():
    """
    Load FAISS vectorstore from Neo4j database.
    """
    with driver.session() as session:
        result = session.run("MATCH (n) WHERE id(n) = 0 RETURN n.embeddings AS embeddings, n.text_chunks AS text_chunks")
        record = result.single()
        if record is None:
            st.error("No embeddings found in Neo4j.")
            return None

        index_str = record['embeddings']
        texts = record['text_chunks']

        # Deserialize FAISS index
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
    """
    Create a conversation chain with memory.
    """
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    """
    Handle user input and log conversation data.
    """
    global conversation_log  # To store conversation data for evaluation

    # Append the user query to the log
    conversation_log["queries"].append(user_question)

    # Get response from the bot
    bot_response = ""
    with driver.session() as session:
        result = session.run("MATCH (n) WHERE n.title =~ '(?i).*release.*' RETURN n")
        for record in result:
            bot_response = record['n']['title']

    if not bot_response:
        response = st.session_state.conversation({'question': user_question})
        bot_response = response['chat_history'][-1].content  # Last message from the bot
        st.session_state.chat_history = response['chat_history']

    # Append the bot response to the log
    conversation_log["responses"].append(bot_response)

    # Save the conversation log and evaluate it after each response
    evaluate_convergence(conversation_log)

    # Display the conversation
    st.write(f"**User**: {user_question}")
    st.write(f"**Bot**: {bot_response}")

def evaluate_convergence(conversation_log):
    """
    Evaluate the convergence of the conversation and save results.
    """
    evaluator = ConvergenceEvaluator()
    evaluator.add_conversation(conversation_log["queries"], conversation_log["responses"])

    # Perform evaluation
    results = evaluator.evaluate()

    # Save results to JSON file
    evaluator.export_results(results, "convergenceResult.json")

def main():
    """
    Main function to run the chatbot with Streamlit interface.
    """
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
