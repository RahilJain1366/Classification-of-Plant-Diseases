from flask import Flask, request, render_template
import os
import numpy as np
import cv2
from keras._tf_keras.keras.models import load_model
import tensorflow as tf
from transformers import ViTImageProcessor, TFViTModel
import joblib

from sklearn.pipeline import make_pipeline

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploaded'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
svm = joblib.load('models/Flask_svm_model.pkl')
class_map = joblib.load('models/Flask_class_map.pkl')
idx_to_class = {v: k for k, v in class_map.items()}

cnn_extractor = load_model('models/Flask_cnn_feature_extractor.h5')
vit_model = TFViTModel.from_pretrained('models/Flask_vit_model')
vit_processor = ViTImageProcessor.from_pretrained('models/Flask_vit_processor')

IMAGE_SIZE = (224, 224)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    array = img.astype('float32')
    preprocessed = tf.keras.applications.resnet50.preprocess_input(array)
    return np.expand_dims(preprocessed, axis=0), img  # for CNN and display

def extract_features(image_np, image_rgb):
    # CNN features
    features_cnn = cnn_extractor.predict(image_np)

    # ViT features
    inputs = vit_processor(images=[image_rgb], return_tensors="tf", do_rescale=False)
    outputs = vit_model(**inputs)
    cls_features = outputs.last_hidden_state[:, 0, :].numpy()

    # Combine
    combined = np.concatenate([features_cnn, cls_features], axis=1)
    return combined

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
