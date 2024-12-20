import json
import logging
import os


def get_file_names(directory):
    files_set = set()
    for root, dirs, files in os.walk(directory):
        for file in files:
            files_set.add(os.path.join(root, file))
    return files_set

def rationale_llama3(model, base_chat=None, output_tokens=1024):
    if not base_chat:
        base_chat = [
            {"role": "system", "content": "You are an AI learning to play chess."},
            {"role": "user", "content": "Teach me the rules of chess. If there's a rule you don't know, ask me."}
        ]

    results = model.invoke(chat=base_chat, max_new_tokens=output_tokens)
    assistant_str = model.parsing(results)
    json_chat = [{base_chat[i]["role"]: base_chat[i]["content"] for i in range(len(base_chat))}]
    json_chat[0] = json_chat[0] | {"assistant": assistant_str}

    if os.path.exists("dataset/training_chess.json"):
        with open('dataset/training_chess.json', 'r') as f:
            json_chat = json.load(f) + json_chat
    with open('dataset/training_chess.json', 'w') as f:
        json.dump(json_chat, f, indent=4)
    json_file = ['dataset/training_chess.json']
    ret_code = model.send_file(json_file, "dataset")
    if ret_code:
        logging.info(f"Send file {json_file} success")
    else:
        logging.warning(f"Send file {json_file} fail")
    return True
