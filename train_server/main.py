import ast
import logging
import multiprocessing
import os
import time
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from model.llama3 import LLama3

logging.getLogger().setLevel(logging.INFO)

class AsyncTrainServer:
    def __init__(self):
        load_dotenv()
        self.event = multiprocessing.Event()
        self.server = os.getenv("MAIN_SERVER")

    def download_file(self):
        file_list_page = urljoin(self.server, "/status-file/dataset")
        file_response = requests.get(file_list_page)
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
                    with open(filename.replace("*", "/"), "wb") as file:
                        file.write(download_response.content)
                self.event.set()
                return True
        return False

    def func_monitor_data(self):
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

    @staticmethod
    def get_file_names(directory):
        files_set = set()
        for root, dirs, files in os.walk(directory):
            for file in files:
                files_set.add(os.path.join(root, file))
        return files_set

    def train_llama3(self, llm_model):
        current_files = self.get_file_names(llm_model.model_output)
        llm_model.train()
        new_files = self.get_file_names(llm_model.model_output) - current_files
        if new_files:
            ret_code = llm_model.send_file(new_files, "model")
            if ret_code:
                logging.info(f"Send file {new_files} success")
                for new_file in new_files:
                    try:
                        os.remove(new_file)
                    except OSError:
                        logging.error(f"Failed to remove {new_file}")
            else:
                logging.warning(f"Send file {new_files} fail")

    def func_llama3(self):
        try:
            i = 0
            json_file = "dataset/training_chess.json"
            model_output = "LLM_models/llama3/model_outputs"
            llm_model = LLama3(json_file=json_file, mode="train", model_output=model_output)
            while True:
                self.event.wait()
                logging.info("train model start")
                self.train_llama3(llm_model)
                logging.info(f"Epoch {i} success")
                i += 1
                time.sleep(3)
        except KeyboardInterrupt:
            logging.info(f"Model train stopped")


    def run(self):
        print(f"Starting train model...")
        process1 = multiprocessing.Process(target=self.func_llama3)
        process2 = multiprocessing.Process(target=self.func_monitor_data)
        process1.start()
        process2.start()
        process1.join()
        process2.join()

        print(f"Finished train model!")


if __name__ == '__main__':
    train_module = AsyncTrainServer()
    train_module.run()
