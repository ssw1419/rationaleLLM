import json
import logging
import os
import random
import sys

import pandas as pd
from trl.commands.cli import train

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from model.llama3 import LLama3

json_file = "dataset/parsing_dataset.json"
model_output = "LLM_models/llama3/parser_LLM/json"

def make_dataset():
    dataset = list()
    llm_model = LLama3(json_file=json_file, mode="test")
    # parsing list
    for i in range(1000):
        data = dict()
        data["system"] = "You are an AI that parses input."
        list_data = list()
        # english dictionary
        df = pd.read_csv("dataset/english.csv")
        for _ in range(random.randint(1,10)):
            list_data.append(df["sentence"][random.randint(0, len(df["sentence"]) - 1)])
        base_chat = [
            {"role": "system", "content": "Do not modify the list_data, only add sentences at the beginning and end."},
            {"role": "user", "content": f"list_data: {list_data}"}
        ]

        results = llm_model.invoke(chat=base_chat)
        appended_list = llm_model.parsing(results)

        data["user"] = appended_list
        data["assistant"] = list_data
        dataset.append(data)
        print(f"-------------Data {i} generated---------------")
        print(data)

    with open(json_file, "w") as f:
        json.dump(dataset, f, indent=4)

def make_dataset_json():
    dataset = list()
    llm_model = LLama3(json_file=json_file, mode="test")
    with open("dataset/parsing_data/json_dataset.json", "r") as f:
        json_data = json.load(f)
    for i in range(len(json_data)):
        data = dict()
        data["system"] = "You are an AI that parses input."
        base_chat = [
            {"role": "system", "content": "You must not modify the json_data, only add some sentences at the beginning and end. Creative sentences are also nice. Never include the instructions I gave in the sentence."},
            {"role": "user", "content": f"json_data: {json_data[i]}"}
        ]

        results = llm_model.invoke(chat=base_chat)
        appended_json = llm_model.parsing(results)

        data["user"] = str(appended_json)
        data["assistant"] = str(json_data[i])
        dataset.append(data)
        print(f"-------------Data {i} generated---------------")
        print(data)

    with open(json_file, "w") as f:
        json.dump(dataset, f, indent=4)


def train_model():
    llm_model = LLama3(json_file=json_file, mode="train", model_output=model_output)
    logging.info("train model start")
    llm_model.train()

def train_lora_model(json_path):
    llm_model = LLama3(json_file=json_path, mode="train", pretrained_model="LLM_models/llama3/parser_LLM/checkpoint-625", model_output=model_output)
    for name, param in llm_model.model.named_parameters():
        print(name)
    logging.info("train model start")
    llm_model.train_lora()


if __name__ == "__main__":
    # json_file = "dataset/parsing_dataset.json"
    train_lora_model("dataset/parsing_data/parsing_dataset_json.json")
