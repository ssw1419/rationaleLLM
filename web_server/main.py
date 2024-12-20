from fastapi import FastAPI
from web_server.src import file_management

app = FastAPI()

@app.get('/')
async def Home():
    return "welcome home"

file_management(app)

# uvicorn web_server.main:app --host 0.0.0.0 --port 8000 --reload