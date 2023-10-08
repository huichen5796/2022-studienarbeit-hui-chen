from fastapi import FastAPI, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import json
import urllib.parse
import uvicorn


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

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)