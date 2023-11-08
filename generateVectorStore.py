import os
import openai
import sys
from dotenv import load_dotenv

dotenv_path = os.path.join('./', 'local.env')
load_dotenv(dotenv_path)
openai.api_key = os.getenv("OPENAI_API_KEY")

# 1.1 Importing files =================================================================

import os

pdf_directory_path = "./docs/"
if os.path.exists(pdf_directory_path):
    files = os.listdir(pdf_directory_path)
    print("Files in the directory:")
    for file in files:
        print(file)
else:
    print(f"The directory '{pdf_directory_path}' does not exist.")
    
from langchain.document_loaders import PyPDFDirectoryLoader

loader = PyPDFDirectoryLoader(pdf_directory_path)
pages = loader.load()

# 1.2 Split Docs into chunk =================================================================

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    length_function=len
)

splits = text_splitter.split_documents(pages)
len(splits)

# 1.3 Convert text chunks to embeddings =================================================================

# Import necessary functionality to perform embeddings and store it in a vectorstore
from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()

import numpy as np
from langchain.vectorstores import Chroma
# Define directory in Google Drive to store the vectors
persist_directory = './vectordb'

# Perform embeddings and store the vectors in the above directory in G Drive
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

# Print the number of vectors stored
print(vectordb._collection.count())