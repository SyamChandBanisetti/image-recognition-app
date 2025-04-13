from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model once
model = ResNet50(weights='imagenet')

def load_and_prep_image(img: Image.Image) -> np.ndarray:
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def predict_image(img: Image.Image):
    processed = load_and_prep_image(img)
    preds = model.predict(processed)
    decoded = decode_predictions(preds, top=3)[0]
    return decoded
