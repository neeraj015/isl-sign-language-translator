# Helper Functions
import numpy as np
import cv2
import tensorflow as tf
import joblib

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="../models/best_model.tflite")
interpreter.allocate_tensors()

# Get input/output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load label encoder
label_encoder = joblib.load("../logs/label_encoder.pkl")

def preprocess_frame(frame, target_size=(64, 64)):
    frame = cv2.resize(frame, target_size)
    frame = frame / 255.0
    return np.expand_dims(frame.astype(np.float32), axis=0)

def predict_from_frame(frame):
    input_data = preprocess_frame(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    class_id = np.argmax(preds)
    label = label_encoder.inverse_transform([class_id])[0]
    confidence = float(preds[class_id])
    return label, confidence
