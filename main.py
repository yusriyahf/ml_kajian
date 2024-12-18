from app.face_detection import register_face, recognize_face, load_embeddings
from app.gender_classification import predict_gender as pred_gender
from fastapi import FastAPI, File, UploadFile
import shutil
import os
import tensorflow as tf
import numpy as np
from PIL import Image
app = FastAPI()





@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # create folder if not exist
    if not os.path.exists('tmp'):
        os.makedirs('tmp')

    # Log untuk memverifikasi file yang diterima
    print(f"Received file: {file.filename} with content type: {file.content_type}")

    # check if the file is an image
    if file.content_type.split('/')[0] != 'image':
        return {"predict": "Invalid file type"}
    
    # Save the uploaded file to the tmp directory
    file_location = f"tmp/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Log untuk memastikan file sudah tersimpan
    print(f"File saved at {file_location}")

    # Perform prediction
    result = recognize_face(file_location, load_embeddings(), 0.6)

    # Log untuk memverifikasi hasil prediksi
    print(f"Prediction result: {result}")

    # Delete the file after prediction
    os.remove(file_location)

    return {"predict": result}


@app.post("/register-face")
async def register(file: UploadFile = File(...), name: str = None):
    # create folder if not exist
    if not os.path.exists('tmp'):
        os.makedirs('tmp')

    # Log untuk memverifikasi file yang diterima
    print(f"Received file: {file.filename} with content type: {file.content_type}")

    # check if the file is an image
    if file.content_type.split('/')[0] != 'image':
        return {"predict": "Invalid file type"}
    
    # Save the uploaded file to the tmp directory
    file_location = f"tmp/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Log untuk memastikan file sudah tersimpan
    print(f"File saved at {file_location}")

    # Perform registration
    register_face(name, file_location)
    
    # Log setelah pendaftaran selesai
    print(f"Face registered successfully for {name}")

    # Delete the file after prediction
    os.remove(file_location)

    return {"message": "Face registered successfully"}



@app.post("/predict-gender")
async def predict_gender(file: UploadFile = File(...)):
    # create folder if not exist
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    # check if the file is an image
    if file.content_type.split('/')[0] != 'image':
        return {"predict": "Invalid file type"}
    # Save the uploaded file to the tmp directory
    file_location = f"tmp/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Perform prediction
    result = pred_gender(file_location)
    # Delete the file after prediction
    os.remove(file_location)
    return {"predict": result}





