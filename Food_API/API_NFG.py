from starlette.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from io import BytesIO
from typing import Tuple

app = FastAPI()

# Allow all origins for CORS (you might want to restrict this in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model = tf.keras.models.load_model('your_model.h5')
# The class names of the model
CLASS_NAMES = ['fish_and_chips', 'french_toast', 'fried_calamari', 'garlic_bread', 'grilled_salmon', 'hamburger', 'ice_cream', 'lasagna', 'macaroni_and_cheese', 'macarons']

@app.get("/")
async def read_root():
    return {"Hello": "World"}


def read_file_as_image(data) -> Tuple[np.ndarray, Tuple[int, int]]:
    img = Image.open(BytesIO(data)).convert('RGB')
    img_resized = img.resize((224, 224))#, resample=Image.BICUBIC)
    image = np.array(img_resized)
    image = np.array(image) / 255.0
    return image, img_resized.size




@app.post("/predict")
async def predict(file: UploadFile = File(...)): # The function that will be executed when the endpoint is called
    try: # A try block to handle any errors that may occur
        image, img_size = read_file_as_image(await file.read()) # Read the image file
        img_batch = np.expand_dims(image, 0) # Add an extra dimension to the image so that it matches the input shape of the model

        predictions = model.predict(img_batch) # Make a prediction
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])] # Get the predicted class
        confidence = np.max(predictions[0]) # Get the confidence of the prediction

        return { # Return the prediction
            'class': predicted_class,
            'confidence': float(confidence)
        }
    except Exception as e: # If an error occurs
        raise HTTPException(status_code=400, detail=str(e)) # Raise an HTTPException with the error message
