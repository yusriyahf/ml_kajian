import numpy as np
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
from PIL import Image
from app.face_detection import extract_face
import os
    
model = load_model('gender_model.h5')

def predict_gender(image_path, target_size = (64, 64)):
    face = extract_face(image_path)
    if face is None:
        return "No face detected"
    
    temp_image_path = "temp_face_image.jpg"
    face_image = Image.fromarray(face)
    face_image.save(temp_image_path)
    
    
    imge = image.load_img(temp_image_path, target_size=target_size)
    X = image.img_to_array(imge)
    X = np.expand_dims(X, axis=0)

    images = np.vstack([X])
    classes = model.predict(images, batch_size=1)
    
    prediction = int(str(classes[0][0])[0])

    # Remove the temporary file
    os.remove(temp_image_path)
    
    if prediction > 0.5:
        print("This is a male")
        return "Laki-laki"
    else:
        print( "This  is a female")
        return "Perempuan"
