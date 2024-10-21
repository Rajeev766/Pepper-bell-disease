import os
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

IMAGE_SIZE = 256

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = "../saved_models/model.keras"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model directory not found: {model_path}")
model = tf.keras.models.load_model(model_path)

class_names = ['Bacterial spot', 'Healthy']

def read_image(data) -> np.array:
    image = Image.open(BytesIO(data))
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.array(image)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Bell Pepper Disease Classification API!"}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    try:
        image = read_image(await file.read())
        image_batch = np.expand_dims(image, axis=0)
        predictions = model.predict(image_batch)
        predicted_class = class_names[np.argmax(predictions[0])]
        accuracy = np.max(predictions[0])
        result = {
            'class': predicted_class,
            'accuracy': float(accuracy)
        }
        return result
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=5000)
