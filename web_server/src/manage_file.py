import os
import shutil

from fastapi.responses import FileResponse
from fastapi import File, UploadFile, Form


class ManageFile:
    def __init__(self):
        self.process = {"model": False, "dataset": False}
        self.files = {"model": set(), "dataset": set()}

    async def download_file(self, filename: str):
        filename = filename.replace("*", "/")
        return FileResponse(path=filename, filename=os.path.basename(filename))

    async def save_file(self, file: UploadFile, msg: str):
        dir_path = os.path.dirname(file.filename)
        os.makedirs(dir_path, exist_ok=True)
        with open(file.filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        if msg in ["model", "dataset"]:
            self.process[msg] = True
            self.files[msg].add(file.filename.replace("/","*"))

    async def process_status(self):
        return self.process


def file_management(app):
    manage = ManageFile()

    @app.post("/upload-file/")
    async def upload_file(file: UploadFile = File(...), message: str = Form(...)):
        await manage.save_file(file, message)
        return {"filename": file.filename}

    @app.get("/download-file/{filename}")
    async def download_file(filename: str, file_type: str = "None"):
        if file_type not in ["model", "dataset"]:
            return []
        if filename in manage.files[file_type]:
            manage.files[file_type].remove(filename)
            if not manage.files[file_type]:
                manage.process[file_type] = False
            return await manage.download_file(filename)
        return []

    @app.get("/status-file/{message}")
    async def status_file(message: str):
        await manage.process_status()
        if message in ["model", "dataset"] and manage.process[message]:
            return manage.files[message]
        return set()
