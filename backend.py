import os
from dotenv import load_dotenv  # Load environment variables from a .env file

import textwrap  # For formatting text if needed later

# Import various libraries used in the script
import langchain
import chromadb
import transformers
import openai
import torch
import requests
import json

# Import specific classes and functions from libraries
from transformers import AutoTokenizer
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM

# =============================================================================
# 1. Environment Setup and API Key Loading
# =============================================================================

# Load environment variables from the .env file (e.g., API keys)
load_dotenv()

# Retrieve tokens from environment variables
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")

# =============================================================================
# 2. Initialize the Language Model with Ollama
# =============================================================================

# Here, we use a locally pulled model called "deepseek-r1" through OllamaLLM.
# This model will be used later in our RetrievalQA chain.
model = OllamaLLM(model="deepseek-r1")

# =============================================================================
# 3. Document Preprocessing Function
# =============================================================================

def docs_preprocessing_helper(file):
    """
    Helper function to load and preprocess a PDF file containing data.

    This function performs two main tasks:
      1. Loads the PDF file using PyPDFLoader from LangChain.
      2. Splits the loaded documents into smaller text chunks using CharacterTextSplitter.
    
    Args:
        file (str): Path to the PDF file.
        
    Returns:
        list: A list of document chunks ready for embedding and indexing.
    """
    # Load the PDF file using LangChain's PyPDFLoader.
    loader = PyPDFLoader(file)
    docs = loader.load_and_split()
    
    # Create a text splitter that divides the documents into chunks up to 5000 characters
    # with an overlap of 200 characters between chunks.
    text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=800)
    docs = text_splitter.split_documents(docs)
    
    return docs

# Preprocess the PDF file and store the document chunks in 'docs'.
docs = docs_preprocessing_helper('dopamine-detox-a-short-guide-to-remove-distractions-and-get-your-brain-to-do-hard-things-9798525995178_compress.pdf')

# =============================================================================
# 4. Set Up the Embedding Function and Chroma Database
# =============================================================================

# Initialize the embedding function from OpenAI. This converts text into numerical vectors.
# The OpenAIEmbeddings class uses the openai_api_key for authentication.
embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Create a vector store (ChromaDB) from the document chunks using the embedding function.
# The Chroma database will be used to retrieve the most relevant documents based on the query.
db = Chroma.from_documents(docs, embedding_function, persist_directory="my_chroma_db")
db.persist()

# =============================================================================
# 5. Define and Initialize the Prompt Template
# =============================================================================

# Define a prompt template that instructs the chatbot on how to answer queries.
# The template includes context information and instructs the bot to use only provided data.
template = """You are a teaching chatbot. Use only the source data provided to answer.

If the answer is not in the source data or is incomplete, say:
"I’m sorry, but I couldn’t find the information in the provided data."

{context}

"""

# Create a PromptTemplate object from LangChain with the defined template.
# It expects a variable called "context" that can be filled later.
prompt = PromptTemplate(template=template, input_variables=["context"])

# Format the prompt with a general context message.
# This additional context tells the chatbot the scenario in which it will be answering questions.
formatted_prompt = prompt.format(
    context="You are interacting with college students. They will ask you questions related to the file provided. Please answer their specific questions using the provided file."
)

# Define a refine prompt for iterative refinement if new context is provided.
refine_prompt_template = """You are a teaching chatbot. We have an existing answer: 
{existing_answer}

We have the following new context to consider:
{context}

Please refine the original answer if there's new or better information. 
If the new context does not change or add anything to the original answer, keep it the same.

If the answer is not in the source data or is incomplete, say:
"I’m sorry, but I couldn’t find the information in the provided data."

Question: {question}

Refined Answer:
"""

refine_prompt = PromptTemplate(
    template=refine_prompt_template,
    input_variables=["existing_answer", "context", "question"]
)

# =============================================================================
# 6. Create the RetrievalQA Chain
# =============================================================================

# The RetrievalQA chain combines:
#   - The language model (model) to generate responses.
#   - A retriever (db.as_retriever) that fetches relevant document chunks based on the query.
#   - A prompt that provides instructions on how to answer the query.
chain_type_kwargs = {
    "question_prompt": prompt,
    "refine_prompt": refine_prompt,
    "document_variable_name": "context",
}

chain = RetrievalQA.from_chain_type(
    llm=model,  # The language model (OllamaLLM with deepseek-r1)
    chain_type="refine",  # "refine" iteratively improves the answer based on additional context.
    retriever=db.as_retriever(search_kwargs={"k": 5}),  # Retrieve the top 5 relevant documents.
    chain_type_kwargs=chain_type_kwargs,
)

# =============================================================================
# 7. Query the Chain and Output the Response
# =============================================================================

# Define a query related to the PDF content.
query = "Cultivate the here-and-now neurotransmitters"

# Run the chain with the query. The chain will:
#   1. Retrieve the most relevant document chunk(s) from ChromaDB.
#   2. Format the prompt with that context.
#   3. Use the language model to generate an answer based on the prompt and retrieved data.
response = chain.run(query)

# Print the response to the console.
print(response)