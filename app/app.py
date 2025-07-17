import streamlit as st
import numpy as np
import tensorflow as tf
import av
import joblib
import requests
import cv2
from pathlib import Path
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

# ────────────── Config ──────────────
st.set_page_config(page_title="ISL Translator", layout="wide")
st.title(" ISL Real-Time Sign Language Translator")

# ────────────── Load Model ──────────────
project_root = Path(__file__).resolve().parent.parent
model_path = project_root / "models" / "best_model.tflite"
label_encoder_path = project_root / "logs" / "label_encoder.pkl"

interpreter = tf.lite.Interpreter(model_path=str(model_path))
interpreter.allocate_tensors()
label_encoder = joblib.load(label_encoder_path)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ────────────── Sidebar ──────────────
st.sidebar.header(" Controls")
enable_tts = st.sidebar.checkbox("🔊 Enable Text-to-Speech")

# ────────────── Session State ──────────────
st.session_state.setdefault("last_label", "")
st.session_state.setdefault("current_label", "")
st.session_state.setdefault("confidence", 0.0)

# ────────────── Preprocess & Predict ──────────────
def preprocess_frame(frame):
    img = cv2.resize(frame, (64, 64))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def predict_from_frame(frame):
    img = preprocess_frame(frame)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    pred_index = np.argmax(preds)
    label = label_encoder.inverse_transform([pred_index])[0]
    confidence = float(preds[pred_index])
    return label, confidence

# ────────────── Get TURN Credentials ──────────────
def get_turn_credentials():
    try:
        account_sid = st.secrets["TWILIO"]["account_sid"]
        auth_token = st.secrets["TWILIO"]["auth_token"]

        response = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Tokens.json",
            auth=(account_sid, auth_token)
        )
        response.raise_for_status()
        data = response.json()
        return data["ice_servers"]
    except Exception as e:
        st.warning(" Could not fetch TURN credentials. Using STUN only.")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

rtc_config = RTCConfiguration({"iceServers": get_turn_credentials()})

# ────────────── Video Processor ──────────────
class SignProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label, confidence = predict_from_frame(img_rgb)
        st.session_state["current_label"] = label
        st.session_state["confidence"] = confidence

        if enable_tts and confidence > 0.90 and label != st.session_state["last_label"]:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.say(f"The sign is {label}")
                engine.runAndWait()
                st.session_state["last_label"] = label
            except:
                pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ────────────── Start WebRTC Stream ──────────────
webrtc_ctx = webrtc_streamer(
    key="isl",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_config,
    video_processor_factory=SignProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# ────────────── Display Result ──────────────
if st.session_state["current_label"]:
    st.markdown(
        f"<h3> Interpreted Sign: <span style='color:#007ACC;'>{st.session_state['current_label']}</span></h3>"
        f"<h4>Confidence: <span style='color:#444;'>{st.session_state['confidence']:.2f}</span></h4>",
        unsafe_allow_html=True
    )
