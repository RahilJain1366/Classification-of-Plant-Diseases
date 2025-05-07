from flask import Flask, request, render_template
import os
import numpy as np
import cv2
import json
import joblib
import tensorflow as tf
from keras._tf_keras.keras.models import load_model
from transformers import ViTImageProcessor, TFViTModel
from keras._tf_keras.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploaded'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ------------------ LOAD MODELS ------------------

print("Loading SVM model...")
svm = joblib.load('models/test_svm_model.pkl')

print("Loading class map...")
with open('models/test_class_map.json', 'r') as f:
    class_map = json.load(f)
# Do not convert keys; assume class_map is like {"American Bollworm on Cotton": 0, ...}
# Invert the mapping: the SVM returns numeric labels, and we need to map them to class names.
idx_to_class = {v: k for k, v in class_map.items()}

print("Loading CNN feature extractor...")
cnn_extractor = load_model('models/Flask_cnn_feature_extractor.h5')

print("Loading ViT model and processor...")
vit_model = TFViTModel.from_pretrained('models/Flask_vit_model')
vit_processor = ViTImageProcessor.from_pretrained('models/Flask_vit_processor')

IMAGE_SIZE = (224, 224)

# ------------------ IMAGE PROCESSING ------------------

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    array = img.astype('float32')
    preprocessed = preprocess_input(array)
    return np.expand_dims(preprocessed, axis=0), img  # (batch, h, w, c), and RGB image

def extract_features(image_np, image_rgb):
    # CNN feature extraction
    features_cnn = cnn_extractor.predict(image_np)

    # ViT feature extraction
    inputs = vit_processor(images=[image_rgb], return_tensors="tf", do_rescale=False)
    outputs = vit_model(**inputs)
    cls_features = outputs.last_hidden_state[:, 0, :].numpy()

    # Concatenate both feature vectors
    combined = np.concatenate([features_cnn, cls_features], axis=1)
    return combined

# ------------------ FLASK ROUTES ------------------

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
