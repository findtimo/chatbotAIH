from main import qa, llm, retriever, custom_prompt
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import csv
from uuid import uuid4

# Testing LLM Evaluation --------------------------------
# 1.Creating dataset
from langsmith import Client
from langchain.smith import RunEvalConfig, run_on_dataset


dataset_name = "Dataset-1"
# project_name = "llmchain-test-a-1"

example_inputs=[]

#1. Read the csv file
with open('testdata/test_data1.csv', 'r', newline='', encoding='windows-1252') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        question, answer = row
        example_inputs.append((question, answer))

#1. Create dataset
client = Client()
# dataset_name = dataset_name #Need to change dataset number each time
dataset = client.create_dataset(
    dataset_name=dataset_name, description="Healthserve Chatbot for AIH Test Data",
)
for input_prompt, expected_answer in example_inputs:
    client.create_example(
        inputs={"question": input_prompt, "chat_history": ""},
        outputs={"answer": expected_answer},
        dataset_id=dataset.id,
    )