import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_mistralai import ChatMistralAI
from langchain.schema import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from pinecone import Pinecone as pc
from langchain_pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from tqdm.autonotebook import tqdm
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain import PromptTemplate
import pandas as pd
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_community.agent_toolkits import create_sql_agent
from dotenv import load_dotenv

load_dotenv()

def retrieve_tool(index, topic, description, pinecone_key=os.getenv("PINECONE_API_KEY")):
    # Initialize Pinecone client
    pc_client = pc(api_key=pinecone_key)
    Index = pc_client.Index(index)

    # Initialize vector store
    vectorstore = Pinecone(Index, embedding=MistralAIEmbeddings())
    retriever = vectorstore.as_retriever(k=1)

    # Create and return the retriever tool
    retriever_tool = create_retriever_tool(
        retriever,
        topic,
        description
    )

    return retriever_tool

## Tool 1
retrieve_tool_1 = retrieve_tool("user-body-measurement-index", 
                                topic="User-Body-Measurements-Retrieval", 
                                description="This tool retrieves user body measurements from the user-body-measurement-index. It can find users with similar or specific measurements or ranges, or retrieve all measurements for a given user ID. Use this when you need to access or analyze user body data for size recommendations based on measurments or user id or customer profiling.",
                                )

## Tool 2
retrieve_tool_2 = retrieve_tool("purchase-history-index", 
                                topic="Purchase-History-Analysis", 
                                description = "This tool accesses the purchase-history-index to retrieve and analyze user purchase data. It can find purchase patterns, return/exchange size history, and successful size choices for individual users. Use this when you need insights into buying behavior, size preferences, or to validate size recommendations."
                                )
## Tool 3
retrieve_tool_3 = retrieve_tool("size-chart-index", 
                                topic="Size-Chart-Generation-and-Retrieval", 
                                description = "This tool interacts with the size-chart-index to generate, update, or retrieve size charts for different product categories. It can provide comprehensive size information including measurements for different sizes (S, M, L, XL, etc.) along with product category and gender. Use this when you need to create or access size charts for specific product categories or to provide size recommendations to users."
                                )

tool = [retrieve_tool_1, 
        retrieve_tool_2, 
        retrieve_tool_3, 
        ]