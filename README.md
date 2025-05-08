# Classification of Plant Diseases

![Plant Disease Classification](https://img.shields.io/badge/AI-Plant%20Disease%20Classification-brightgreen)
![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey)

A robust plant disease classification system using a hybrid **CNN-SVM** architecture to identify diseases in plant leaves from images.

## 📋 Project Overview

This project implements a machine learning solution to identify and classify various plant diseases from leaf images. The system leverages:

- **Convolutional Neural Networks (CNN)** for feature extraction
- **Support Vector Machines (SVM)** for classification
- **Flask** for web application deployment

## 🌿 Dataset

The complete dataset for training and testing is available for download:
- [Download Plant Disease Dataset](https://utdallas.box.com/s/nakpwnwuh7yprafdatb1geu4vxv8oy2n)

## ✨ Features

- **Accurate Disease Classification**: Identify plant diseases with high accuracy
- **Hybrid CNN-SVM Architecture**: Combines the feature extraction capabilities of CNNs with the classification power of SVMs
- **User-Friendly Interface**: Upload plant leaf images through a simple web interface
- **Real-Time Predictions**: Get instant disease classification results
- **Scalable Design**: Architecture can be expanded to identify more plant diseases

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/RahilJain1366/Classification-of-Plant-Diseases.git
   cd Classification-of-Plant-Diseases
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create necessary directories**
   ```bash
   mkdir -p models static/upload
   ```

5. **Run the application**
   ```bash
   python <python scripts starting with app>
   ```

6. **Access the web interface**
   - Open your browser and go to: http://127.0.0.1:5000/

## 📁 Project Structure

```
Classification-of-Plant-Diseases/
│
├── app.py                # Flask application entry point
├── requirements.txt      # Python dependencies
├── models/               # Trained model files
├── static/               # Static files (CSS, JS, images)
│   └── upload/           # Uploaded images storage
└── templates/            # HTML templates for the web interface
```

## 🖥️ Usage

1. Navigate to the web interface at http://127.0.0.1:5000/
2. Upload an image of a plant leaf
3. Click "Submit" to get the disease classification results

## 🔬 Model Architecture

The classification system uses a hybrid approach:
- **CNN**: Extracts relevant features from plant leaf images
- **SVM**: Uses these features to classify the disease with high accuracy

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please reach out to rahiljain1366@gmail.com

---

*Note: This project is for educational purposes and should not replace professional agricultural advice.*
