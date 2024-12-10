from app.face_detection import register_face, recognize_face, load_embeddings
from app.gender_classification import predict_gender as pred_gender
from fastapi import FastAPI, File, UploadFile
import shutil
import os
import tensorflow as tf
import numpy as np
from PIL import Image
app = FastAPI()

gender_model = tf.keras.models.load_model('gender_detection_model_terbaru.h5')

def preprocess_image(image_path):
    """Preproses gambar untuk model deteksi gender."""
    img = Image.open(image_path).convert('RGB')  # Membuka gambar dan mengubah ke RGB
    img = img.resize((96, 96))  # Ubah ukuran gambar menjadi (96, 96) seperti yang diharapkan model
    img_array = np.array(img) / 255.0  # Normalisasi nilai piksel ke rentang [0, 1]

    # Pastikan gambar memiliki bentuk yang benar (96, 96, 3)
    img_array = np.expand_dims(img_array, axis=0)  # Menambahkan dimensi batch (1, 96, 96, 3)

    return img_array



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



@app.post("/check-gender")
async def check_gender(file: UploadFile = File(...)):
    if not os.path.exists('tmp'):
        os.makedirs('tmp')

    print(f"Received file: {file.filename} with content type: {file.content_type}")

    if file.content_type.split('/')[0] != 'image':
        return {"message": "Invalid file type"}
    
    file_location = f"tmp/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"File saved at {file_location}")

    try:
        # Preprocess the image for the model
        img_array = preprocess_image(file_location)

        # Verifikasi bentuk gambar
        print(img_array.shape)  # Harus mencetak (1, 96, 96, 3)

        # Predict the gender using the model
        prediction = gender_model.predict(img_array)
        print(f"Prediction: {prediction}")

        # Assuming the model outputs a single value between 0 and 1 (e.g., Male: 0, Female: 1)
        gender = "Male" if prediction[0][0] < 0.5 else "Female"
        print(f"Gender prediction: {gender}")

    except Exception as e:
        print(f"Error during gender prediction: {e}")
        return {"message": "Error during prediction, please try again later."}
    finally:
        os.remove(file_location)

    return {"gender": gender}


