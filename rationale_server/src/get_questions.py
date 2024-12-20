import ast
import json
import os
import re
import time

import praw

import requests
from dotenv import load_dotenv
from googleapiclient.discovery import build
from newsapi import NewsApiClient

load_dotenv()

def get_questions_llama3(model, output_tokens=1024):
    for _ in range(3):
        base_chat = [
            {"role": "system", "content": "You are an AI learning to play chess."},
            {"role": "user", "content": "What more data do you need to learn to play chess?"}
        ]

        results = model.invoke(chat=base_chat, max_new_tokens=output_tokens)
        assistant_str = model.parsing(results)
        print(f"***** first: {assistant_str}")
        base_chat = [
            {"role": "system", "content": "Please provide a short keyword search to get the information below. Be sure to include a list of “queries=[]” at the end of your answer."},
            {"role": "user", "content": assistant_str}
        ]
        results = model.invoke(chat=base_chat, max_new_tokens=output_tokens)
        assistant_str = model.parsing(results)
        print(f"***** second: {assistant_str}")

        base_chat = [
            {"role": "system", "content": "Answer by extracting only one list from the answer. The answer must start and end with '[', ']'."},
            {"role": "user", "content": assistant_str}
        ]
        for _ in range(3):
            results = model.invoke(chat=base_chat, max_new_tokens=output_tokens)
            assistant_str = model.parsing(results)
            print(f"***** third: {assistant_str}")
            try:
                assistant_str = ast.literal_eval(assistant_str)
                if type(assistant_str[0]) == str:
                    break
            except Exception as e:
                print(f"Error: {e}")
        if type(assistant_str) == list and type(assistant_str[0]) == str:
            break
    if type(assistant_str) == list and type(assistant_str[0]) == str:
        return assistant_str
    return []

def find_index(directory):
    files = os.listdir(directory)
    numbers = []
    for file in files:
        match = re.search(r'^\d+$', file)
        if match:
            numbers.append(int(file))
    current = 0
    if numbers:
        current = max(numbers) + 1
    current = str(current)
    current = '0' * (5-len(current)) + current
    return current

def google_search(query, api_key, cse_id, num_results=10):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id, num=num_results).execute()
    return res['items'] if 'items' in res else []

def via_google(model, output_tokens=1024):
    # Google Custom Search API
    query = get_questions_llama3(model, output_tokens=output_tokens)
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    results = google_search(query, api_key, cse_id)
    current = find_index("dataset/google_search")
    with open(f"dataset/google_search/{current}.json", "w") as out:
        json.dump(results, out, indent=4)
    return results

def via_nytimes(model, output_tokens=1024):
    queries = get_questions_llama3(model, output_tokens=output_tokens)
    api_key = os.getenv("NYTIMES_API_KEY")
    response = ""
    for query in queries:
        request_url = f"https://api.nytimes.com/svc/search/v2/articlesearch.json?q={query}&begin_date=20230101&end_date=20241201&&api-key={api_key}"
        response = requests.get(request_url)
        if response.status_code == 200:
            data = response.json()
            print(data)
            with open(f"dataset/nytimes_search/{query.lower()}.json", "w") as out:
                json.dump(data, out, indent=4)
        else:
            print(f"Request failed with status code {response.status_code}")
        break
    return response

def via_newsapi(model, output_tokens=1024):
    newsapi = NewsApiClient(api_key=os.getenv("NEWSAPI_KEY"))
    # queries = get_questions_llama3(model, output_tokens=output_tokens)
    queries = ["popular chess openings"]
    api_key = os.getenv("NYTIMES_API_KEY")
    response = ""
    for query in queries:
        response = newsapi.get_everything(q=query,
                                          from_param='2024-01-01',
                                          to='2024-12-12',
                                          language='en',
                                          sort_by='relevancy')
        print(response)
        with open(f"dataset/newsapi_search/{query.lower()}.json", "w") as out:
            json.dump(response, out, indent=4)

def via_reddit(model, output_tokens=1024):
    queries = get_questions_llama3(model, output_tokens=output_tokens)
    user_agent = os.getenv("REDDIT_USER_AGENT")
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_ID"),
        client_secret=os.getenv("REDDIT_API_KEY"),
        user_agent=user_agent,
    )
    reddit_dir = "dataset/reddit_search"
    queries_file = os.path.join(reddit_dir, "queries.json")
    if os.path.exists(queries_file):
        with open(queries_file, "r") as f:
            searched_queries = json.load(f)
    else:
        searched_queries = []
    for query in queries:
        if "chess" not in query.lower():
            query = "chess: " + query
        if query.lower() in searched_queries:
            continue
        searched_queries.append(query.lower())
        for submission in reddit.subreddit('all').search(query, limit=2, time_filter='month'):
            files = list()
            current = find_index(reddit_dir)
            data_dir = os.path.join(reddit_dir, current)
            body = submission.selftext if submission.is_self else ""
            base_chat = [
                {"role": "system", "content": "Is it related to chess? Answer only with 'Yes' or 'No'."},
                {"role": "user", "content": f"{body[:8192]}"}
            ]
            response = model.invoke(base_chat, output_tokens)
            response = model.parsing(response)
            if "No" in response:
                continue
            os.makedirs(data_dir, exist_ok=True)
            filename = os.path.join(data_dir, f"body.json")
            with open(filename, "w") as out:
                json.dump(body, out, indent=4)
            files.append(filename)
            submission.comments.replace_more(limit=None)
            for j, comment in enumerate(submission.comments.list()):
                if comment.body == "[deleted]":
                    continue
                filename = os.path.join(data_dir, f"comment_{j}.json")
                with open(filename, "w") as out:
                    json.dump(comment.body, out, indent=4)
                files.append(filename)
            aggregate_dataset(files, model)
            with open(queries_file, "w") as f:
                json.dump(searched_queries, f, indent=4)
    return [f"dataset/training_chess.json"]

def aggregate_dataset(files, model):
    aggregate_data = []
    json_data = ""
    for file in files:
        with open(file, "r") as fp:
            json_data += json.load(fp)
    while len(json_data) > 8192:
        new_json_data = ""
        for j in range(0, len(json_data), 8192):
            base_chat = [{"role": "system", "content": "You are an AI explaining chess. Summarize only the content related to chess strategy from the input data."},
                         {"role": "user", "content": json_data[j:j + 16384]}]
            response = model.invoke(base_chat, max_new_tokens=4096)
            response = model.parsing(response)
            new_json_data += response
        json_data = str(new_json_data)
    base_chat = [{"role": "system",
                  "content": "You are an AI explaining chess. Answer only the content related to chess strategy from the input data."},
                 {"role": "user", "content": json_data}]
    response = model.invoke(base_chat, max_new_tokens=8192)
    response = model.parsing(response)

    base_chat = {"system": "You are an AI explaining chess.",
                 "user": "Let the community know what chess users think.",
                 "assistant": response}
    aggregate_data.append(base_chat)
    with open(f"dataset/training_chess.json", "r") as out:
        original_data = json.load(out)
    original_data += aggregate_data
    with open(f"dataset/training_chess.json", "w") as out:
        json.dump(original_data, out, indent=4)
