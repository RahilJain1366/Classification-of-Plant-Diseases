import os
import numpy as np
import cv2
import json
import joblib
import tensorflow as tf

from flask import Flask, request, render_template
from keras._tf_keras.keras.applications import ResNet50
from keras._tf_keras.keras.applications.resnet50 import preprocess_input
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.preprocessing.image import img_to_array

from transformers import ViTImageProcessor, TFViTModel

# ------------------ CONFIG ------------------
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploaded'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

IMAGE_SIZE = (224, 224)

# ------------------ LOAD MODELS ------------------

print("Loading SVM model...")
svm = joblib.load('models/test_svm_model.pkl')

print("Loading class map...")
with open('models/test_class_map.json', 'r') as f:
    class_map = json.load(f)
idx_to_class = {v: k for k, v in class_map.items()}

print("Loading CNN (ResNet50) feature extractor...")
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
cnn_extractor = Model(inputs=resnet_model.input, outputs=resnet_model.output)

print("Loading ViT model and processor...")
vit_model = TFViTModel.from_pretrained('models/Flask_vit_model')
vit_processor = ViTImageProcessor.from_pretrained('models/Flask_vit_processor')

# ------------------ IMAGE PROCESSING ------------------

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    array = img_to_array(img)
    preprocessed = preprocess_input(array)  # For ResNet
    return np.expand_dims(preprocessed, axis=0), img  # (batch, h, w, c), RGB img

def extract_features(image_np, image_rgb):
    # CNN features
    features_cnn = cnn_extractor.predict(image_np)

    # ViT features
    inputs = vit_processor(images=[image_rgb], return_tensors="tf", do_rescale=False)
    outputs = vit_model(**inputs)
    cls_features = outputs.last_hidden_state[:, 0, :].numpy()

    # Concatenate both
    combined = np.concatenate([features_cnn, cls_features], axis=1)
    return combined

# ------------------ ROUTES ------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            image_np, image_rgb = preprocess_image(filepath)
            features = extract_features(image_np, image_rgb)
            pred = svm.predict(features)[0]
            prediction = idx_to_class[pred]
            filename = file.filename

    return render_template('index.html', prediction=prediction, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
