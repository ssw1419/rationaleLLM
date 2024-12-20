import ast
import json
import logging
import multiprocessing
import os
import time
from urllib.parse import urljoin
import requests
from dotenv import load_dotenv
import torch

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from src.make_rationale import rationale_llama3
from src.get_questions import via_google, via_reddit
from model.llama3 import LLama3

class AsyncRationaleServer:
    def __init__(self, pretrained_model="LLM_models/llama3/original",
                 json_file="../dataset/training_chess.json"):
        load_dotenv()
        self.event = multiprocessing.Event()
        self.server = os.getenv("MAIN_SERVER")
        self.json_file = json_file
        self.pretrained_model = pretrained_model
        self.model = ""

    def download_file(self):
        file_list_page = urljoin(self.server, "/status-file/model")
        file_response = requests.get(file_list_page)
        # print(f"Request dataset page: {file_response}")
        if file_response.status_code == 200 and file_response.text:
            filenames = ast.literal_eval(file_response.text)
            if len(filenames) != 0:
                print(f"download_file start!!")
                self.event.clear()
                for filename in filenames:
                    dir_path = os.path.dirname(filename.replace("*", "/"))
                    os.makedirs(dir_path, exist_ok=True)
                    download_page = urljoin(self.server, f"download-file/{filename}?file_type=dataset")
                    download_response = requests.get(download_page)
                    with open(filename.replace("*", "/") + ".test", "wb") as file:
                        file.write(download_response.content)
                self.event.set()
                return True
        return False

    def func_monitor_model(self):
        try:
            i = 0
            self.event.set()
            while True:
                ret_code = self.download_file()
                if ret_code:
                    logging.info(f"Download data {i} success")
                    i += 1
                time.sleep(3)
        except KeyboardInterrupt:
            logging.info(f"Data monitor stopped")
        return 0

    def func_llama3(self):
        try:
            self.model = LLama3(json_file=self.json_file, mode="test", pretrained_model=self.pretrained_model)
            i = 0
            while True:
                self.event.wait()
                print("func_llama3", i)
                ret_code = rationale_llama3(self.model)
                if not ret_code:
                    continue
                logging.info(f"Epoch {i} success")
                i += 1
                if True:
                    files = via_reddit(rationale_module.model, 1024)
                    self.model.send_file(files, "dataset")
                time.sleep(3)
        except KeyboardInterrupt:
            logging.info(f"Model rationale stopped")

    def run(self):
        print(f"Starting rationale model...")
        process1 = torch.multiprocessing.Process(target=self.func_llama3)
        process2 = torch.multiprocessing.Process(target=self.func_monitor_model)
        process1.start()
        process2.start()
        process1.join()
        process2.join()
        print(f"Finished rationale model!")

    def monitor_model(self):
        pass

if __name__ == '__main__':
    rationale_module = AsyncRationaleServer(pretrained_model="LLM_models/llama3/parser_LLM/json/checkpoint-740")
    rationale_module.run()
