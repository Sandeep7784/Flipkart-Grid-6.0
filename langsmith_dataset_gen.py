import csv
from langsmith import Client
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN")

client = Client()

# Define dataset
dataset_name = "Body"
dataset = client.create_dataset(dataset_name)

def determine_size(gender, height, weight, bust_chest, waist, hips):
    # Convert height to inches
    height_parts = height.split("'")
    height_inches = int(height_parts[0]) * 12 + int(height_parts[1].strip('"'))
    
    if gender == "Female":
        if bust_chest <= 32 and waist <= 24 and hips <= 34:
            return "XS"
        elif bust_chest <= 34 and waist <= 26 and hips <= 36:
            return "S"
        elif bust_chest <= 36 and waist <= 28 and hips <= 38:
            return "M"
        elif bust_chest <= 38 and waist <= 30 and hips <= 40:
            return "L"
        elif bust_chest <= 40 and waist <= 32 and hips <= 42:
            return "XL"
        elif bust_chest <= 42 and waist <= 34 and hips <= 44:
            return "2XL"
        else:
            return "2XL+"
    else:  # Male
        if bust_chest <= 34 and waist <= 28:
            return "XS"
        elif bust_chest <= 36 and waist <= 30:
            return "S"
        elif bust_chest <= 38 and waist <= 32:
            return "M"
        elif bust_chest <= 40 and waist <= 34:
            return "L"
        elif bust_chest <= 42 and waist <= 36:
            return "XL"
        elif bust_chest <= 44 and waist <= 38:
            return "2XL"
        else:
            return "2XL+"

# Read and process the CSV data
inputs = []
outputs = []

# Read from the CSV file
# with open('test_data.csv', 'r', newline='') as csvfile:
# with open('test_data.csv', 'r', newline='') as csvfile:
df = pd.read_csv('test_data.csv')

# csv_reader = csv.DictReader(csvfile)
df = df.head(5)
for _, row in df.iterrows():
    size = determine_size(row["Gender"], row["Height"], float(row["Weight"]), 
                            float(row["Bust/Chest"]), float(row["Waist"]), float(row["Hips"]))
    
    # Generate input in the requested format
    inputs.append({
        "question": f"What is the size for a {row['Gender']} with height {row['Height']}, weight {row['Weight']}, "
                    f"bust/chest {row['Bust/Chest']}, waist {row['Waist']}, and hips {row['Hips']}?"
    })
    
    # Generate output in the requested format
    outputs.append({
        "answer": f"The size is {size}."
    })
# print(inputs[1])
# print(outputs[1])
# Create examples in the dataset
client.create_examples(
    inputs=inputs,
    outputs=outputs,
    dataset_id=dataset.id,
)