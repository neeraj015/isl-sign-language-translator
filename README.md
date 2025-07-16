#  Indian Sign Language Translator (ISL) – Real-Time Web App

This project is a **real-time Indian Sign Language (ISL) to text & speech translator** that uses deep learning (CNN), TensorFlow Lite, and Streamlit. It works on both desktop and mobile, supports webcam input, and gives voice + on-screen predictions for each ISL gesture.

<div align="center">
  <img src="https://github.com/yourusername/isl-sign-language-translator/assets/demo.gif" width="600"/>
</div>

---

## Features

-  **Live ISL Gesture Recognition** via webcam
-  **Text-to-Speech (TTS)** output (desktop)
-  **Confidence score** for every prediction
-  **Mobile & Desktop Compatible**
-  **Custom-trained CNN** with 99% accuracy
-  **Lightweight TFLite Model** for fast inference
-  Rejects predictions with low confidence
-  Webcam ON/OFF controls and camera selector
-  Expandable: LSTM, "No Gesture" class, MediaPipe Hands, full-sentence predictions, etc.

---

##  Project Structure

isl-sign-language-translator/
│
├── app/ # Streamlit app
│ └── app.py
│
├── data/ # Dataset
│ └── raw/ # Organized gesture folders + no_gesture/
│ └── processed/ # Preprocessed arrays (.npy)
│
├── models/ # Saved models (.h5, .tflite)
├── logs/ # label encoder, prediction outputs
│
├── notebooks/ # Jupyter notebooks (EDA, Training, GradCAM)
│ ├── 01_data_preprocessing.ipynb
│ ├── 02_model_training.ipynb
│ ├── 03_gradcam.ipynb 
│ ├── 04_model_evaluation.ipynb
│
├── README.md
└── requirements.txt


