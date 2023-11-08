from datetime import datetime, timedelta
import os
from langsmith import Client
from typing import Optional
from langchain.load.load import load
from langchain.schema import get_buffer_string
import pandas as pd

client = Client()

#https://docs.smith.langchain.com/evaluation/capturing-feedback
# LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
# LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"

PROJECT_NAME = "evaluators"

start_time = datetime.utcnow() - timedelta(days=7)

runs = list(client.list_runs(
    project_name=PROJECT_NAME,
    run_type="llm",
    error=False,
    # start_time=start_time,
))

import re

def extract_input_section(text):
    # Define a regular expression pattern to match the input section
    pattern = r'\[Input\]:+([^*]+)\*\*\*+'

    # Search for the pattern in the text
    match = re.search(pattern, text)

    if match:
        input_section = match.group(1).strip()
        return input_section
    else:
        return ""
    
def extract_submission_section(text):
    pattern = r'\[Submission\]:+([^*]+)\*\*\*+'

    # Search for the pattern in the text
    match = re.search(pattern, text)

    if match:
        input_section = match.group(1).strip()
        return input_section
    else:
        return ""
    
    
def stringify_inputs(inputs: dict) -> dict:
    # print(inputs.keys())
    messages = inputs.get('messages')
    if messages is None:
        return {
            "messages": "",
        }
    
    return {
        "messages": extract_input_section(get_buffer_string(load(inputs['messages'])))
    }
    
def stringify_submission(inputs: dict) -> dict:
    # print(inputs.keys())
    messages = inputs.get('messages')
    if messages is None:
        return {
            "submission": "",
        }
    
    return {
        "submission": extract_submission_section(get_buffer_string(load(inputs['messages'])))
    }

def stringify_outputs(outputs: Optional[dict]) -> dict:
    if not outputs:
        return {}
    if 'generations' not in outputs:
        return {
            "generated_message": ""
        }
    if isinstance(outputs['generations'], dict):
        return {
            "generated_message": get_buffer_string([load(outputs['generations']['message'])])
        }
    else:
        return {
            "generated_message": get_buffer_string([load(outputs['generations'][0]['message'])])
        }

from functools import lru_cache
from langsmith.schemas import Run

@lru_cache(maxsize=1000)
def _fetch_run(run_id: str) -> Run:
    return client.read_run(run_id)

def get_root(run: Run) -> Run:
    if run.execution_order == 1:
        return run
    return _fetch_run(str(run.parent_run_ids[-1]))

def get_feedback(r: Run) -> dict:
    if not r.feedback_stats:
        return {}
    
    print(r.feedback_stats.items())
    return {k: v['avg'] for k, v in r.feedback_stats.items()}

def get_root_feedback(run: Run) -> dict:
    root_run = get_root(run)
    return get_feedback(root_run)

df = pd.DataFrame(
    [{
        **stringify_inputs(run.inputs),
        **stringify_submission(run.inputs),
        # **run.inputs,
        **stringify_outputs(run.outputs),
        **get_root_feedback(run),
        # "error":run.error,
        # "latency": (run.end_time - run.start_time).total_seconds() if run.end_time else None, # Pending runs have no end time
        # "prompt_tokens": run.prompt_tokens,
        # "completion_tokens": run.completion_tokens,
        # "total_tokens": run.total_tokens,
    }
          for run in runs
    ],
    index=[run.id for run in runs]
)

csv_file_path = "output.csv"

df.to_csv(csv_file_path, index=False)
# print(client.read_project(project_name=PROJECT_NAME,).feedback_stats)
# print(client.read_project(project_name=chain_results["project_name"]).feedback_stats)