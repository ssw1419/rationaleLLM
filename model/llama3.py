import logging
import os
from urllib.parse import urljoin
from typing import Literal

import requests
from datasets import Dataset
from dotenv import load_dotenv
import json
from peft import LoraConfig, PeftModel, get_peft_model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer, SFTConfig

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from .LLM_prompt import llama_prompt_txt

load_dotenv()

class LLama3:
    load_dotenv()
    def __init__(self, json_file, mode:Literal["train", "test"],
                 pretrained_model="LLM_models/llama3/original",
                 model_output="model_dataset/llama3/model_outputs",
                 lora_config = None, bnb_config = None, model_config = None):
        self.access_token = os.getenv("HUGGINGFACE_API_KEY")
        if not lora_config:
            lora_config = {"r": 8, "lora_alpha": 8, "task_type": "CAUSAL_LM",
                           "target_modules": ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]}
        self.lora_config = LoraConfig(**lora_config)
        if not bnb_config:
            bnb_config = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.float16}
        self.bnb_config = BitsAndBytesConfig(**bnb_config)
        abspath = os.getcwd()
        if "server" in abspath:
            abspath = os.path.join(os.path.dirname(abspath), pretrained_model)
        else:
            abspath = os.path.join(abspath, pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(abspath, token=self.access_token)
        if not model_config:
            model_config = {"pretrained_model_name_or_path": abspath,
                            "device_map": "auto", "torch_dtype": torch.bfloat16, "quantization_config": self.bnb_config,
                            "attn_implementation": "eager", "use_cache": False}
        self.model = AutoModelForCausalLM.from_pretrained(**model_config)
        self.json_file = json_file
        self.server = os.getenv("MAIN_SERVER")
        if mode == "train":
            self.dataset = self.json_to_dataset(self.json_file)
            self.model_output = model_output
            self.train_config = {"output_dir": self.model_output, "num_train_epochs": 20, "per_device_train_batch_size": 2,
                                 "gradient_accumulation_steps": 8, "optim": "paged_adamw_8bit",
                                 "gradient_checkpointing": True, "gradient_checkpointing_kwargs": {'use_reentrant':False},
                                 "learning_rate": 2e-5, "bf16": True, "logging_steps": 20,
                                 "push_to_hub": False, "save_steps": 100, "max_seq_length": 1024}

    def json_to_dataset(self, json_file):
        eos_token = self.tokenizer.eos_token
        with open(json_file) as f:
            data = json.load(f)
        dataset = dict()
        dataset["text"] = []
        for i in range(len(data)):
            system_message = data[i]["system"]
            human_message = data[i]["user"]
            assistant_message = data[i]["assistant"]
            dataset["text"].append(llama_prompt_txt.format(system_message, human_message, assistant_message) + eos_token)
        return Dataset.from_dict(dataset)

    def update_train_config(self, train_config):
        self.train_config = train_config

    def update_model(self, train_model_path):
        self.model = PeftModel.from_pretrained(self.model, model_id=train_model_path)


    def train(self, train_config=None):
        if train_config:
            self.train_config = train_config
        training_params = SFTConfig(**self.train_config)
        trainer = SFTTrainer(
            model=self.model,
            args=training_params,
            train_dataset=self.dataset,
            peft_config=self.lora_config,
            packing=False,
        )
        trainer.train()

    def train_lora(self, train_config=None):
        self.model = get_peft_model(self.model, self.lora_config)

        for name, param in self.model.named_parameters():
            if "lora" not in name:
                param.requires_grad = False

        if train_config:
            self.train_config = train_config
        training_params = SFTConfig(**self.train_config)
        trainer = SFTTrainer(
            model=self.model,
            args=training_params,
            train_dataset=self.dataset,
            peft_config=self.lora_config,
            packing=False,
        )
        trainer.train()


    def invoke(self, chat:list[dict], max_new_tokens=256) -> str:
        prompt = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_dict=True,
                                                    return_tensors="pt").to(self.model.device)
        input_ids = prompt["input_ids"]
        attention_mask = prompt["attention_mask"]
        outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                      pad_token_id=self.tokenizer.eos_token_id,
                                      max_new_tokens=max_new_tokens, do_sample=True)

        output_message = self.tokenizer.decode(outputs[0])

        return output_message


    def send_file(self, files:list, message:Literal["model", "dataset"]):
        for file_path in files:
            with open(file_path, 'rb') as f:
                file_dict = {'file': (file_path, f)}
                data = {"message": message}
                response = requests.post(urljoin(self.server, "/upload-file/"), files=file_dict, stream=True, data=data)
            if response.status_code != 200:
                logging.warning(f"response failed: {response.status_code}, {file_path}")
                return False
        return True

    @staticmethod
    def parsing(response):
        assistant_chk = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        if assistant_chk not in response:
            return ""
        assistant_index = response.find(assistant_chk)
        assistant_str = response[assistant_index + len(assistant_chk):]
        assistant_str = assistant_str.replace("<|eot_id|>", "")
        return assistant_str


if __name__ == '__main__':
    model = LLama3(json_file="./dataset/training_chess.json", model_output="./model_dataset/llama3/model_outputs2")
    model.train()
