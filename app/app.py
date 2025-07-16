import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import joblib
from pathlib import Path

# ────────────── Streamlit Config ──────────────
st.set_page_config(page_title="ISL Translator", layout="wide")
st.title("🤟 ISL Real-Time Sign Language Translator")

# ────────────── Load TFLite Model & Labels ──────────────
project_root = Path(__file__).resolve().parent.parent
model_path = project_root / "models" / "best_model.tflite"
label_encoder_path = project_root / "logs" / "label_encoder.pkl"

interpreter = tf.lite.Interpreter(model_path=str(model_path))
interpreter.allocate_tensors()
label_encoder = joblib.load(label_encoder_path)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ────────────── Sidebar Controls ──────────────
st.sidebar.header("🔧 Controls")
camera_index = st.sidebar.selectbox("Select Camera", options=[0, 1, 2], index=0)
start_camera = st.sidebar.checkbox("📷 Start Camera")
enable_tts = st.sidebar.checkbox("🔊 Enable Text-to-Speech")

# ────────────── Webcam & UI Setup ──────────────
frame_window = st.empty()
prediction_text = st.empty()  # 👈 NEW: to show prediction on main UI

# ────────────── Session State ──────────────
if "last_label" not in st.session_state:
    st.session_state["last_label"] = ""

# ────────────── Frame Preprocessing ──────────────
def preprocess_frame(frame):
    img = cv2.resize(frame, (64, 64))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# ────────────── Prediction Function ──────────────
def predict_from_frame(frame):
    img = preprocess_frame(frame)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    pred_index = np.argmax(preds)
    label = label_encoder.inverse_transform([pred_index])[0]
    confidence = float(preds[pred_index])
    return label, confidence

# ────────────── Webcam Loop ──────────────
if start_camera:
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        st.error("❌ Failed to access camera.")
    else:
        st.success("✅ Camera started. Uncheck box to stop.")

    while cap.isOpened() and start_camera:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Could not read frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        label, confidence = predict_from_frame(frame_rgb)

        # Show webcam frame
        frame_window.image(frame_rgb)

        # ✅ Show prediction on main screen (NEW!)
        prediction_text.markdown(
            f"<h3>🧠 Interpreted Sign: <span style='color:#007ACC;'>{label}</span></h3>"
            f"<h4>Confidence: <span style='color:#444;'>{confidence:.2f}</span></h4>",
            unsafe_allow_html=True
        )

        # Show prediction in sidebar
        st.sidebar.markdown(f"**Predicted Label:** `{label}`")
        st.sidebar.markdown(f"**Confidence:** `{confidence:.2f}`")

        # 🔊 Speak only if label changed and confidence is high
        if enable_tts and confidence > 0.90 and label != st.session_state["last_label"]:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.say(f"The sign is {label}")
                engine.runAndWait()
                st.session_state["last_label"] = label
            except Exception as e:
                st.warning(f"TTS failed: {e}")

    cap.release()
    st.success("🛑 Camera stopped.")
