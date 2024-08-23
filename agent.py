import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_mistralai import ChatMistralAI
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain import PromptTemplate
from tools import tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from langsmith.client import LangSmithClient
from langgraph import LangGraphClient

# Initialize LangSmith
langsmith_client = LangSmithClient(api_key=os.getenv("LANGCHAIN"))

# Initialize LangGraph
langgraph_client = LangGraphClient(api_key=os.getenv("LANGCHAIN"))
load_dotenv()

store = {}
os.environ["MISTRAL_API_KEY"] = os.getenv('MISTRAL_API_KEY')
os.environ["HF_TOKEN"] = os.getenv("HUGGING_FACE_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN")

# Initialize language model
llm = ChatMistralAI(model="mistral-large-latest", streaming=True)

# Bind the tool to the model
llm = llm.bind_tools(tool)

def log_and_store_response(question, response):
    # Log the interaction using LangSmith
    langsmith_client.log_interaction(
        question=question,
        response=response,
        metadata={"model": "ChatMistralAI"}
    )

    # Optionally, create a visualization or summary
    visualization = langgraph_client.create_visualization(
        data=response,  # Assuming the response includes tabular or structured data
        chart_type="table"  # You can specify other chart types like "bar", "line", etc.
    )
    
    return visualization

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Define agent
agent = create_tool_calling_agent(llm, tool, hub.pull("hwchase17/openai-functions-agent"))
agent_executor = AgentExecutor(agent=agent, tools=tool)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def agent_call(question):
    def create_prompt_template():
        template = """
        As an AI-powered size chart generator for apparel sellers, your primary responsibility is to generate accurate size charts and provide size recommendations based on user data and purchase history.

        **Instructions**:
        - Use the 'User-Body-Measurements-Retrieval' tool to access and analyze user body measurement data.
        - Utilize the 'Purchase-History-Analysis' tool to retrieve and analyze user purchase data, including returns and exchanges.
        - Employ the 'Size-Chart-Generation-and-Retrieval' tool to generate, update, or retrieve size charts for different product categories.
        - Cluster similar body types and their corresponding successful purchases to improve recommendations.
        - Generate comprehensive size charts including measurements for different sizes (S, M, L, XL, etc.) where weight is in 'kg' unit.
        - Provide confidence scores for each measurement in the generated size chart.
        - Ensure the system can handle different apparel categories (e.g., tops, bottoms, dresses).
        - Focus on reducing size-related returns by providing accurate recommendations.
        - Adapt recommendations and size charts for new brands or product lines as needed.
        - Prioritize processing speed and efficiency in generating and updating size charts.

        **Prompt Structure**:
        ```
        Question: {question}
        ```

        **Response Guidelines**:
        - Tailor responses based on the user's body measurements and purchase history.
        - Provide clear, concise, and helpful size recommendations.
        - Structure responses with bullet points for clarity and ease of reading.
        - Include confidence scores for size recommendations when applicable.
        - Highlight important information about fit and sizing.
        - If generating a size chart, ensure it's comprehensive and easy to understand.
        - Explain any adjustments made based on the user's purchase history or return data.
        - If recommending a size different from the user's usual choice, provide a clear rationale.
        - Do not mention specific tools used in the response.

        **Important**: Focus solely on providing helpful and informative size recommendations or generating accurate size charts. Exclude any tool invocation commands from the response text.
        """
        return PromptTemplate.from_template(template=template)

    def format_prompt(prompt_template, question):
        return prompt_template.format(
            question=question,
        )

    # Prepare the prompt
    prompt_template = create_prompt_template()
    formatted_prompt = format_prompt(prompt_template, question)

    # Invoke the agent synchronously
    response = agent_with_chat_history.invoke({"input": formatted_prompt}, {"configurable": {"session_id": "<foo>"}})
    visualization = log_and_store_response(question, response['output'])

    return response['output'], visualization

question = input("Ask a question:")
response, visualization = agent_call(question)
print(response)
print("Visualization URL:", visualization['url']) 
