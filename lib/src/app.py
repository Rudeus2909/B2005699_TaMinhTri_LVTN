import asyncio
import shutil
import uuid
from typing import Any, Dict
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import logging
from src.dl_code.predict_image import single_predict
import os
from contextlib import asynccontextmanager


logger = logging.getLogger(__name__)
app = FastAPI()

origins = [
    "http://127.0.0.1:5500",  
    "http://localhost:5500"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Mã chạy khi ứng dụng khởi động
    logger.info("Startup: Initializing application")
    yield  # Đây là nơi các yêu cầu sẽ được xử lý
    # Mã chạy khi ứng dụng tắt
    logger.info("Shutdown: Cleaning up")

app.router.lifespan_context = lifespan
@app.get("/health")
async def health(make_error: bool = None) -> Dict[Any, Any]:
    if make_error:
        raise Exception("test problem.")
    return {}

@app.post("/api/v1/predict", tags=["Predict"])
async def api_predict(upload_file: UploadFile = File(...)) -> dict:
    print("Go here")
    file_name = upload_file.filename
    print(file_name)
    ext = file_name.split(".")[-1]
    allow_ext = ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]
    if ext not in allow_ext:
        return {
            "status": 500,
            "message": "The file is not in the correct format"
        }
    file_name = "static/image/" + str(uuid.uuid4()) + ".jpg"
    print(file_name)
    with open(file_name, 'wb') as image_save:
        shutil.copyfileobj(upload_file.file, image_save)

    result = await single_predict(file_name)
    print(result)
    return {
        "status": 200,
        "data": result
    }