import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from deepface import DeepFace
from mtcnn.mtcnn import MTCNN
from PIL import Image
from numpy import asarray

model = DeepFace.build_model("Facenet")

def get_embedding(image_path):
    img = load_img(image_path, target_size=(160, 160))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    embedding = model.model.predict(img_array)
    return embedding[0]

def save_embedding(name, embedding, file_path="embeddings.pkl"):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
    else:
        data = {}
    
    data[name] = embedding
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    
    print(f"Embedding for {name} saved successfully.")

def load_embeddings(file_path="embeddings.pkl"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Embedding file not found at {file_path}")
    
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
    print(f"Loaded embeddings: {list(data.keys())}")  # Log nama-nama yang terdaftar
    return data


def compare_embeddings(embedding1, embedding2):
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity[0][0]


def extract_face(filename, required_size=(160, 160)):
	image = Image.open(filename)
	image = image.convert('RGB')
	pixels = asarray(image)
	detector = MTCNN()
	results = detector.detect_faces(pixels)

	if len(results) == 0:
		return None
	x1, y1, width, height = results[0]['box']

	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height

	face = pixels[y1:y2, x1:x2]

	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

def register_face(name, image_path):
    face_array = extract_face(image_path)
    if face_array is None:
        print(f"No face detected in image: {image_path}")
        return "No face detected"
    
    temp_image_path = "temp_face_image.jpg"
    face_image = Image.fromarray(face_array)
    face_image.save(temp_image_path)

    embedding = get_embedding(temp_image_path)
    print(f"Generated embedding for {name}: {embedding[:5]}...")  # Log sebagian embedding untuk memverifikasi

    save_embedding(name, embedding)

    # Remove the temporary file
    os.remove(temp_image_path)

def recognize_face(image_path, embeddings, threshold=0.5):
    face_array = extract_face(image_path)
    if face_array is None:
        print(f"No face detected in image: {image_path}")
        return "No face detected"
    
    temp_image_path = "temp_face_image.jpg"
    face_image = Image.fromarray(face_array)
    face_image.save(temp_image_path)

    # Get embedding from the temporary file
    embedding = get_embedding(temp_image_path)
    print(f"Generated embedding for prediction: {embedding[:5]}...")  # Log sebagian embedding untuk memverifikasi

    # Remove the temporary file
    os.remove(temp_image_path)

    max_similarity = -1
    recognized_name = None
    for name, registered_embedding in embeddings.items():
        similarity = compare_embeddings(embedding, registered_embedding)
        print(f"Comparing {name}: similarity = {similarity}")  # Log perbandingan similarity

        if similarity > max_similarity:
            max_similarity = similarity
            recognized_name = name

    if max_similarity < threshold:
        return "Unknown"
    return recognized_name
