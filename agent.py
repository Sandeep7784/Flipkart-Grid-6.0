import sys
import os
from langchain_mistralai import ChatMistralAI
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain import PromptTemplate
from tools import tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from langsmith.evaluation import LangChainStringEvaluator, evaluate
from langsmith.schemas import Run, Example
from tenacity import retry, stop_after_attempt, wait_exponential, time
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
load_dotenv()

# Environment setup
store = {}
os.environ["MISTRAL_API_KEY"] = os.getenv('MISTRAL_API_KEY')
os.environ["HF_TOKEN"] = os.getenv("HUGGING_FACE_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN")

# Initialize language model
llm = ChatMistralAI(model="mistral-large-latest", streaming=True)
llm = llm.bind_tools(tool)

# Evaluation setup
_PROMPT_TEMPLATE = """You are an expert professor specialized in grading students' answers to questions.
You are grading the following question:
{query}
Here is the real answer:
{answer}
You are grading the following predicted answer:
{result}
Respond with CORRECT or INCORRECT:
Grade:
"""
PROMPT = PromptTemplate(
    input_variables=["query", "answer", "result"], template=_PROMPT_TEMPLATE
)
eval_llm = ChatMistralAI(temperature=0.0)
qa_evaluator = LangChainStringEvaluator("qa", config={"llm": eval_llm, "prompt": PROMPT})

def evaluate_length(run: Run, example: Example) -> dict:
    prediction = run.outputs.get("output") or ""
    required = example.outputs.get("answer") or ""
    score = int(len(prediction) < 2 * len(required))
    return {"key":"length", "score": score}

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

def create_prompt_template():
    template = """
    You are an AI-powered size chart generator for apparel sellers. Your task is to answers to size-related questions based on the given measurements.

    Available Tools:
    1. 'User-Body-Measurements-Retrieval': use for user data
    2. 'Purchase-History-Analysis': use for purchase history
    3. 'Size-Chart-Generation-and-Retrieval': use for size charts

    Instructions:
    1. Use these tools to generate accurate size recommendations: 'User-Body-Measurements-Retrieval', 'Purchase-History-Analysis', 'Size-Chart-Generation-and-Retrieval'. 
    2. Analyze the provided measurements and compare them to standard size charts.
    3. Provide a clear, specific size recommendation (Available sizes S/L/M/XL/XS/XXL).
    4. Consider different apparel categories (e.g., tops, bottoms, dresses) in your recommendation.

    Input:
    Question: {question}

    Response Format:
    1. Provide a precise answer to the user's question.
    2. Start with the recommended size, then briefly explain the rationale.
    3. If applicable, mention any potential fit issues or suggest alternatives.
    4. Use bullet points for clarity if providing multiple pieces of information.
    5. Include a confidence score for your recommendation (e.g., "80% confident").
    6. Keep the response under 100 words unless more detail is absolutely necessary.

    Guidelines: 
    - Always provide a size recommendation based on the information given like height, weight, chest, waist, hips, etc.
    - Do not mention the tools or process used to generate the answer.
    - Focus solely on answering the user's question about sizing.
    """
    return PromptTemplate.from_template(template=template)

def format_prompt(prompt_template, question):
    return prompt_template.format(question=question)

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def agent_call(question):
    prompt_template = create_prompt_template()
    formatted_prompt = format_prompt(prompt_template, question)
    try:
        response = agent_with_chat_history.invoke({"input": formatted_prompt}, {"configurable": {"session_id": "<foo>"}}) 
        return response['output']
    except Exception as e:
        if "429" in str(e):
            print(f"Rate limit exceeded. Retrying in a moment...")
            time.sleep(random.uniform(1, 3))  # Add a random delay between 1 and 3 seconds
            raise  # Re-raise the exception to trigger a retry
        else:
            raise  # If it's not a rate limit error, raise the exception normally

def langsmith_app(inputs):
    output = agent_call(inputs["question"])
    return {"output": output}

# Main execution
if __name__ == "__main__":
    # question = input("Ask a question: ")
    # response = agent_call(question)
    # print(response)

    # Evaluation
    dataset_name = "Body"  
    experiment_results = evaluate(
        langsmith_app,
        data=dataset_name,
        evaluators=[evaluate_length, qa_evaluator],
        experiment_prefix="mistral-large",
    )
    print("Evaluation results:", experiment_results)