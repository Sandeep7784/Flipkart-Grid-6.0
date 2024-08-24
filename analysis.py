from langchain_anthropic import ChatAnthropic
from langchain_core.prompts.prompt import PromptTemplate
from langsmith.evaluation import LangChainStringEvaluator
from langsmith.schemas import Run, Example
import openai
from langsmith.evaluation import evaluate
from langsmith import Client
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set up LangSmith client
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN")
client = Client()

# Define the prompt template
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

# Set up the evaluator
eval_llm = ChatAnthropic(temperature=0.0)
qa_evaluator = LangChainStringEvaluator("qa", config={"llm": eval_llm, "prompt": PROMPT})

# Define the length evaluator function
def evaluate_length(run: Run, example: Example) -> dict:
    prediction = run.outputs.get("output") or ""
    required = example.outputs.get("answer") or ""
    score = int(len(prediction) < 2 * len(required))
    return {"key": "length", "score": score}

# Set up OpenAI client
openai_client = openai.Client()

# Define the main application function
def my_app(question):
    return openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Respond to the users question in a short, concise manner (one short sentence)."
            },
            {
                "role": "user",
                "content": question,
            }
        ],
    ).choices[0].message.content

# Define the LangSmith application function
def langsmith_app(inputs):
    output = my_app(inputs["question"])
    return {"output": output}

# Set the dataset name
dataset_name = "Body Measurements Size Dataset"

# Run the evaluation
experiment_results = evaluate(
    langsmith_app,  # Your AI system
    data=dataset_name,  # The data to predict and grade over
    evaluators=[evaluate_length, qa_evaluator],  # The evaluators to score the results
    experiment_prefix="openai-3.5",  # A prefix for your experiment names to easily identify them
)

# Print or process the experiment results as needed
print(experiment_results)