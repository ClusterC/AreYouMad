import streamlit as st
import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd

# --- (ส่วนโหลดโมเดลและ tokenizer เหมือนเดิม) ---
SAVED_MODEL_PATH = 'saved_model_best'
tokenizer = AutoTokenizer.from_pretrained(SAVED_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(SAVED_MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
labels = {"น้อยใจ": 0, "งอน": 1, "ประชด": 2, "ปกติ": 3}
id2label = {v: k for k, v in labels.items()}
# --- (จบส่วนโหลดโมเดล) ---

# --- (ฟังก์ชัน predict_emotion เหมือนเดิม) ---
def predict_emotion(text):
    """
    ฟังก์ชันสำหรับทำนายอารมณ์จากข้อความ โดยใช้โมเดลที่โหลดมา
    และคืนค่าอารมณ์ที่ทำนายได้พร้อมความน่าจะเป็น (ทั้งแบบ format และ numeric)
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class_id = torch.argmax(probs, dim=-1).item()
    predicted_label = id2label[predicted_class_id]

    probabilities_formatted = {label: f"{prob.item() * 100:.2f}%" for label, prob in zip(labels.keys(), probs[0])}
    probabilities_numeric = {label: prob.item() * 100 for label, prob in zip(labels.keys(), probs[0])}

    return {
        "predicted_label": predicted_label,
        "probabilities_formatted": probabilities_formatted,
        "probabilities_numeric": probabilities_numeric
    }

# --- (response_generator เหมือนเดิม) ---
def response_generator(prompt):
    prediction_result = predict_emotion(prompt)
    response_text = f"ฉันคิดว่าคุณกำลังรู้สึก '{prediction_result['predicted_label']}' นะ"
    for word in response_text.split():
        yield word + " "
        time.sleep(0.05)

# --- (ส่วน Streamlit UI) ---
st.title("Are You Mad Chatbot!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ปรับการแสดงผล History ให้รองรับกราฟ (ใส่ color)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "chart_data" in message and message["chart_data"] is not None:
             # --- แก้ไขตรงนี้: ใส่ color='Emotion' ---
             st.bar_chart(message["chart_data"], x='Emotion', y='Probability (%)', color='Emotion')

# Accept user input
if prompt := st.chat_input("พิมพ์ข้อความที่นี่..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- ส่วนการแสดงผลของ Assistant ---
    with st.chat_message("assistant"):
        prediction_result = predict_emotion(prompt)
        response_stream = response_generator(prompt)
        response_text = st.write_stream(response_stream)

        probs_numeric = prediction_result['probabilities_numeric']
        df_probs = pd.DataFrame(list(probs_numeric.items()), columns=['Emotion', 'Probability (%)'])

        # --- แก้ไขตรงนี้: ใส่ color='Emotion' ---
        st.bar_chart(df_probs, x='Emotion', y='Probability (%)', color='Emotion')

    # เพิ่ม response ของ assistant (ทั้ง text และข้อมูลกราฟ) ลงใน history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "chart_data": df_probs
        })