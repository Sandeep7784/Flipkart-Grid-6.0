__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import time
import csv
import random
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv

load_dotenv()

def document_split(data):
  # Initialize an empty list to store Document objects
  documents = []
  
  # Iterate over the dictionary items
  for user_id, info in data.items():
    # Construct content and metadata for each query
    content = f"User ID: {user_id}\n"
    content += "\n".join([f"{key}: {value}" for key, value in info.items()])

    metadata = {
        "user_id": user_id,
        **info
    }

    documents.append(Document(page_content=content, metadata=metadata))

  # Initialize the text splitter
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

  # Split documents into chunks
  all_splits = text_splitter.split_documents(documents)

  return all_splits

def pinecone_vector_store(data):
  index_name = "purchase-history-index"
  pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

  existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

  if index_name not in existing_indexes:
    pc.create_index(
      name=index_name,
      dimension=1024,
      metric="cosine",
      spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
      time.sleep(1)
  else:
    print("The index already exists so enter a new index name")

  index = pc.Index(index_name)

  docs = document_split(data)

  # Create vector store from documents in batches, to avoid overwhelming the system with too many requests at once.
  batch_size = 100
  for i in range(0, len(docs), batch_size):
    batch = docs[i:i+batch_size]
    PineconeVectorStore.from_documents(batch, embedding=MistralAIEmbeddings(), index_name=index_name)
    print(f"Processed batch {i//batch_size + 1} of {len(docs)//batch_size + 1}")

  print(f"Index '{index_name}' has been created and populated with user body measurements.")

def read_csv(filename):
  data_dict = {}
  allowed_headers = ["user_id", "company_id", "company", "product_category", "size", "gender"]

  with open(filename, mode='r') as file:
    csv_reader = csv.reader(file)
    headers = next(csv_reader)

    for row in csv_reader:
      user_id = row[0]
      data_dict[user_id] = {headers[i]: row[i] for i in range(len(headers)) if headers[i] in allowed_headers}
      data_dict[user_id]['is_size_exchange'] = False

  return data_dict

filename = 'data.csv'

data = read_csv(filename)

pinecone_vector_store(data)