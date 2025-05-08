# Classification of Plant Diseases

This project implements a **plant disease classification system** using a **custom CNN-SVM** architecture. The model is designed to classify images of plant leaves and predict the disease based on their visual features.

## Project Overview

- **Objective**: To classify plant diseases using a hybrid model that combines Convolutional Neural Networks (CNN) for feature extraction and Support Vector Machines (SVM) for classification.
- **Dataset**: The dataset for training and testing is available for download [here](https://utdallas.box.com/s/nakpwnwuh7yprafdatb1geu4vxv8oy2n).

## Features

- **Image Upload**: Users can upload plant leaf images, and the model will predict the disease.
- **Hybrid Architecture**: Combines CNN for feature extraction and SVM for classification, ensuring accuracy and efficiency.
- **Flask App**: The model is served via a Flask application for easy deployment and testing.

## Installation

Follow the steps below to set up the project locally:

### 1. Clone the repository

Clone this repository to your local machine:

```bash
git clone https://github.com/RahilJain1366/Classification-of-Plant-Diseases.git
cd Classification-of-Plant-Diseases

pip install -r requirements.txt

mkdir -p models static/upload

Classification-of-Plant-Diseases/
│
├── app.py                # Flask application entry point
├── requirements.txt      # Python dependencies
├── models/               # Folder for storing model files
├── static/               # Folder for static files (images, uploads)
│   └── upload/           # Folder for storing uploaded images
└── templates/            # HTML templates for the Flask app
