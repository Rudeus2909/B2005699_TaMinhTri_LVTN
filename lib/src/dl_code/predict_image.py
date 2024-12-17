from typing import Any
from fastapi import File, UploadFile
import tensorflow as tf
import keras
import numpy as np
import cv2
import json
import os
import random

def model_load(path_model: str)-> Any:
    keras.backend.clear_session()
    model = keras.models.load_model(path_model)
    return model

def load_image_into_np_arr(data):
    npimg = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

async def single_predict(path_file: str) -> dict:
    print('Đang dự đoán...')
    model = model_load('D:\Graduation\Project\check_point\EfficientNet_model_30e.keras')
    image = cv2.imread(path_file)
    image = cv2.resize(image, dsize=(224, 224))
    
    # Load label dictionary
    with open('json/number_to_label_dict.json', 'r', encoding='utf-8') as f:
        label_dict = json.load(f)

    # Dự đoán xác suất
    predict = model.predict(image.reshape(1, 224, 224, 3))[0]

    # Sắp xếp xác suất giảm dần và lấy top 3
    sorted_indices = np.argsort(predict)[::-1]  # Chỉ số sắp xếp giảm dần
    top_indices = sorted_indices[:3]  # Lấy 3 nhãn đầu
    top_labels = {label_dict[str(i)]: float(predict[i]) for i in top_indices}

    # Lấy hình ảnh ngẫu nhiên từ thư mục của từng nhãn
    image_dir = r'D:\Graduation\Project\Dataset\Validation'  # Thư mục chứa ảnh theo từng nhãn
    label_images = {}

    for label, prob in top_labels.items():
        label_folder = os.path.join(image_dir, label)
        if os.path.exists(label_folder):
            # Lấy 3 ảnh ngẫu nhiên từ thư mục của nhãn
            images = random.sample(
                [os.path.join(label_folder, img) for img in os.listdir(label_folder) if img.endswith(('.jpg', '.png'))],
                min(3, len(os.listdir(label_folder)))
            )
            label_images[label] = images

    print(label_images)
    # Tính tổng xác suất của các loại còn lại
    other_probability = sum(predict[sorted_indices[3:]])
    if other_probability > 0:
        top_labels["Các loại khác"] = other_probability

    return {
        "path_file": path_file,
        "top_labels": top_labels,
        "label_images": label_images
    }