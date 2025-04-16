import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ระบุ path ไปยังโฟลเดอร์ที่บันทึกโมเดลไว้
SAVED_MODEL_PATH = 'saved_model_best' # หรือ path เต็ม เช่น 'C:/path/to/your/saved_model'

# โหลด tokenizer และ model จากโฟลเดอร์ที่บันทึกไว้
tokenizer = AutoTokenizer.from_pretrained(SAVED_MODEL_PATH)
# โหลดโมเดลพื้นฐาน (Hugging Face) ไม่ใช่ LightningModule wrapper
model = AutoModelForSequenceClassification.from_pretrained(SAVED_MODEL_PATH)

# ตั้งค่า device (ใช้ GPU ถ้ามี)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() # ตั้งค่าโมเดลเป็น evaluation mode (สำคัญมาก!)

# สร้างฟังก์ชัน predict (คล้ายกับของเดิม แต่ใช้โมเดลที่โหลดมาใหม่)
labels = {"น้อยใจ": 0, "งอน": 1, "ประชด": 2, "ปกติ": 3}
# สร้าง inverse mapping เพื่อแปลง id กลับเป็น label name (ถ้าต้องการ)
id2label = {v: k for k, v in labels.items()}

def predict_emotion(text):
    """
    ฟังก์ชันสำหรับทำนายอารมณ์จากข้อความ โดยใช้โมเดลที่โหลดมา
    """
    # Tokenize ข้อความ
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # ย้าย input tensors ไปยัง device เดียวกับโมเดล
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ไม่ต้องคำนวณ gradient สำหรับ inference
    with torch.no_grad():
        outputs = model(**inputs)

    # คำนวณความน่าจะเป็น
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class_id = torch.argmax(probs, dim=-1).item()

    # แสดงผลลัพธ์
    print(f"ข้อความ: {text}")
    print(f"อารมณ์ที่ทำนายได้: {id2label[predicted_class_id]}")
    print("ความน่าจะเป็น:")
    for label, prob in zip(labels.keys(), probs[0]):
        print(f"  - {label}: {prob.item() * 100:.2f}%")
    print("-" * 20)

# --- ทดสอบการใช้งาน ---
predict_emotion("เออ ดีมากเลยนะ")
predict_emotion("สวัสดีครับ")
predict_emotion("ทำไมเธอถึงไม่ตอบแชทฉันเลย")
predict_emotion("ฉันไม่สำคัญหรอก")
predict_emotion("ทำไมเธอถึงไม่ตอบแชทฉันเลย ขอโทษนะ ฉันยุ่งอยู่จริงๆ อ๋อ งั้นก็ไม่เป็นไร ฉันคงไม่สำคัญอยู่แล้ว ไม่ใช่แบบนั้นนะ! วันนี้เธอลืมไปใช่ไหม ลืมอะไรเหรอ?")
