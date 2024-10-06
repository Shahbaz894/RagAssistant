import os
from dotenv import load_dotenv
import logging
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from typing import List
import nest_asyncio
from phi.assistant import Assistant
from phi.document import Document
from phi.document.reader.pdf import WebsiteReader
from phi.llm.openai import OpenAIChat
from phi.knowledge import AssistantKnowledge
from phi.tools.duckduckgo import DuckDuckGo
from phi.embedder.openai import OpenAIEmbedder
from phi.vectordb.pgvector import Pgvector2
from phi.storage.assistant.postgres import PgAssistantStorage
import psycopg
import chainlit as cl  # Import Chainlit for app integration

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Set OpenAI API key from .env file

# Database URL
db_url = 'https://blogs.nvidia.com/blog/what-is-a-transformer-model/'  # Assign your database URL here

# Set up logging for debugging purposes
logger = logging.getLogger(__name__)

# Setup function for the Assistant
def setup_assistant(llm: str) -> Assistant:
    """
    Set up the assistant with a language model, vector database for knowledge storage,
    and document embedding using OpenAI's ada-002 model.
    """
    return Assistant(
        name='auto rag assistant',  # Name the assistant
        llm=llm,  # Use the provided language model
        storage=PgAssistantStorage(
            table_name='auto_rag_assistant_openai',  # PostgreSQL table for assistant data
            db_url=db_url  # Database connection
        ),
        knowledge_base=AssistantKnowledge(
            vector_db=Pgvector2(  # Use PgVector for vector storage in Postgres
                db_url=db_url,  # Database connection
                collection='auto_rag_documents_openai',  # Collection of documents
                embedder=OpenAIEmbedder(
                    model='text-embedding-ada-002',  # Model used for creating embeddings
                    apikey=os.getenv("OPENAI_API_KEY")  # API key for embedding model
                )
            ),
            num_documents=3  # Limit the number of documents to search through
        ),
        description="You are a helpful Assistant called 'AutoRAG' and your goal is to assist the user in the best way possible.",  # Assistant's description
        instructions=[  # Instructions that guide the assistant's behavior
            "Given a user query, first ALWAYS search your knowledge base using the `search_knowledge_base` tool to see if you have relevant information.",
            "If you don't find relevant information in your knowledge base, use the `duckduckgo_search` tool to search the internet.",
            "If you need to reference the chat history, use the `get_chat_history` tool.",
            "If the user's question is unclear, ask clarifying questions to get more information.",
            "Carefully read the information you have gathered and provide a clear and concise answer to the user.",
            "Do not use phrases like 'based on my knowledge' or 'depending on the information'.",
        ],
        show_tool_calls=True,  # Show tool usage in responses
        search_knowledge=True,  # Enable knowledge base search
        read_chat_history=True,  # Enable chat history reading
        tools=[DuckDuckGo()],  # Use DuckDuckGo for external searches
        markdown=True,  # Enable markdown support in responses
        add_chat_history_to_messages=True,  # Include chat history in responses
        add_datetime_to_instructions=True,  # Include current datetime in instructions
        debug_mode=True  # Enable debug mode for more detailed logs
    )

# Function to add a document to the assistant's knowledge base
def add_document_to_kb(assistant: Assistant, file_path: str, file_type: str = 'pdf'):
    """
    Add a document to the assistant's knowledge base. Only PDFs are supported by default.
    """
    if file_type == 'pdf':
        reader = WebsiteReader()  # Use WebsiteReader to read PDF documents
    else:
        raise ValueError('Unsupported file type')  # Raise error if unsupported file type is provided
    
    documents: List[Document] = reader.read(file_path)  # Read the PDF file and convert to a Document
    if documents:
        assistant.knowledge_base.load_documents(documents, upsert=True)  # Load documents into the assistant's knowledge base
        logger.info(f"Document '{file_path}' added to the knowledge base.")  # Log success
    else:
        logger.error("Could not read document")  # Log error if document couldn't be read

# Function to query the assistant
def query_assistant(assistant: Assistant, question: str):
    """
    Ask the assistant a question and return the response.
    """
    response = ""
    for delta in assistant.run(question):  # Run the assistant query and gather responses
        response += delta  # type: ignore  # Accumulate the response in chunks
    return response  # Return the full response

# Chainlit integration: function that handles user messages
@cl.on_message
async def on_message(message: str):
    """
    Chainlit message handler. Queries the assistant and returns the response.
    """
    query = message  # User's message
    assistant = setup_assistant(llm)  # Initialize assistant with the LLM
    response = query_assistant(assistant, query)  # Run the assistant query
    await cl.Message(content=response).send()  # Send the response back to Chainlit UI

# Main program entry point
if __name__ == "__main__":
    nest_asyncio.apply()  # Apply nest_asyncio to handle nested event loops
    
    # Fetch the model name from environment variables or default to 'gpt-4'
    llm_model = os.getenv("OPENAI_MODEL_NAME", "gpt-4")
    
    # Initialize OpenAIChat with the model name and API key
    llm = OpenAIChat(model=llm_model, api_key=os.getenv("OPENAI_API_KEY"))  # Fetch API key from environment variable
    
    # Set up the assistant
    assistant = setup_assistant(llm)
    
    # Add a sample PDF document to the assistant's knowledge base
    sample_pdf_path = "data/attention.pdf"
    add_document_to_kb(assistant, sample_pdf_path, file_type="pdf")
    
    # Define a sample query
    query = "What is the main topic of the document?"
    
    # Run the query against the assistant
    response = query_assistant(assistant, query)
    
    # Output the query and the assistant's response
    print("Query:", query)
    print("Response:", response)
