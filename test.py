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
project_name = "llmchain-final-1-2"

client = Client()

# example_inputs=[]
#1. Read the csv file
# with open('testdata/test_data.csv', 'r', newline='', encoding='windows-1252') as csvfile:
#     csvreader = csv.reader(csvfile)
#     for row in csvreader:
#         question, answer = row
#         example_inputs.append((question, answer))

# #2. Create dataset
# # dataset_name = dataset_name #Need to change dataset number each time
# dataset = client.create_dataset(
#     dataset_name=dataset_name, description="Healthserve Chatbot for AIH Test Data",
# )
# for input_prompt, expected_answer in example_inputs:
#     client.create_example(
#         inputs={"question": input_prompt, "chat_history": ""},
#         outputs={"answer": expected_answer},
#         dataset_id=dataset.id,
#     )

eval_config = RunEvalConfig(
  evaluators=[
    # "criteria",
    "qa",
    "context_qa",
    "cot_qa",
    # RunEvalConfig.Criteria("harmfulness"),
  ],
    input_key="question", 
    prediction_key="answer",
)

def chain_constructor():
    new_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key='question',
        output_key='answer'
    )
    
    qa_new = ConversationalRetrievalChain.from_llm(
        llm,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
        memory=new_memory,
        get_chat_history=lambda h : h
    )
    
    return qa_new

chain_results = run_on_dataset(
    client=client,
    dataset_name=dataset_name,
    llm_or_chain_factory=chain_constructor,
    evaluation=eval_config,
    verbose=True,
    project_name=project_name, 
    
)



print(dataset_name + " " + project_name)