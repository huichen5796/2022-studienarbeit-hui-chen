from fastapi import FastAPI, Request, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import json
import urllib.parse
import uvicorn
from pipeline_3 import pipeline_3

from lib import *
from conf import *

import os
import time
import shutil

log_writer = LogWriter()

app = FastAPI()

# Allow cross domain requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return 'success'

@app.post("/fileUpload/")
async def fileUpload(files: list[UploadFile], user:str):
    save_dir = 'store_ori_file' if not user else 'user/store_ori_file'
    res = {}
    for file in files:
        try:
            file_name = os.path.basename(file.filename)
            contents = await file.read()
            with open(f"{save_dir}/{file_name}", "wb") as f:
                f.write(contents)
            res[file_name] = True
        except:
            res[file_name] = False

    return res

websocket_connections = []

@app.websocket("/start")
async def start(websocket: WebSocket, user = None):
    await websocket.accept()
    websocket_connections.append(websocket)

    path_images = open_all(user=user)
    start = time.perf_counter()
    for i, image_path in enumerate(path_images):
        error = ''
        result = {}
        try:
            result = pipeline_3(image_path)
            log_writer.writeSuccess(f'DONE: {image_path}')
        except Exception as e:
            log_writer.writeError(str(e))
            error = str(e)

        finish = '▓' * int((i+1)*(50/len(path_images)))
        need_do = '-' * (50-int((i+1)*(50/len(path_images))))
        dur = time.perf_counter() - start
        message = f"{(i+1)}/{len(path_images)}|{finish}{need_do}|{dur:.2f}s done: {os.path.basename(image_path)} error: {error}"
        data = {
            "bar": message,
            "now": {
                "file_name": os.path.basename(image_path),
                "result": result,
                "error": error
            },
            "finish": i+1 == len(path_images)
        }
        await send_json_to_websocket(websocket, data)

    await websocket.close()

async def send_websocket_message(websocket: WebSocket, message: str):
    await websocket.send_text(message)

async def send_json_to_websocket(websocket: WebSocket, data: dict):
    await websocket.send_json(data)

async def send_websocket_message_to_all(message: str):
    for websocket in websocket_connections:
        await send_websocket_message(websocket, message)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)