import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split # เพิ่มเข้ามา

# --- (ส่วนข้อมูล data และ labels เหมือนเดิม) ---
data = [
    # ----------------------- น้อยใจ -----------------------
    ("ฉันไม่สำคัญหรอก", "น้อยใจ"),
    ("เธอคงไม่สนใจฉันอยู่แล้ว", "น้อยใจ"),
    ("ไม่เป็นไร ฉันชินแล้ว", "น้อยใจ"),
    ("ฉันก็แค่คนที่ไม่สำคัญ", "น้อยใจ"),
    ("จะทำอะไรก็ไม่ต้องบอกฉันก็ได้นะ", "น้อยใจ"),
    ("เธอลืมฉันไปแล้วใช่ไหม", "น้อยใจ"),
    ("ไม่มีฉันก็คงไม่เป็นไรหรอก", "น้อยใจ"),
    ("เธอไปสนใจคนอื่นเถอะ", "น้อยใจ"),
    ("ทำไมฉันต้องเป็นคนสุดท้ายที่รู้ตลอด", "น้อยใจ"),
    ("ฉันคงเป็นตัวเลือกที่ไม่สำคัญ", "น้อยใจ"),
    ("ไม่เป็นไร ฉันอยู่คนเดียวได้", "น้อยใจ"),
    ("ก็แค่ลืมฉันไปเหมือนเดิมแหละ", "น้อยใจ"),
    
    ("ทำไมเธอถึงไม่ตอบแชทฉันเลย", "น้อยใจ"),
    ("ขอโทษนะ ฉันยุ่งอยู่จริงๆ", "ปกติ"),
    ("อ๋อ งั้นก็ไม่เป็นไร ฉันคงไม่สำคัญอยู่แล้ว", "น้อยใจ"),
    ("ไม่ใช่แบบนั้นนะ!", "ปกติ"),

    # ----------------------- งอน -----------------------
    ("ทำอะไรก็ทำไปเลย ไม่สนแล้ว", "งอน"),
    ("ตามใจเธอเลย", "งอน"),
    ("ไม่ต้องมายุ่งกับฉันแล้ว", "งอน"),
    ("โอเค ไม่ต้องพูดอะไรแล้ว", "งอน"),
    ("ก็ดีนะ ฉันไม่ว่าอะไรอยู่แล้ว", "งอน"),
    ("เธอไม่ต้องมาพูดกับฉันอีก", "งอน"),
    ("ไม่เป็นไร ฉันโอเค", "งอน"),
    ("เธอจะไปไหนก็ไปเลย", "งอน"),
    ("อยากทำอะไรก็ทำไป", "งอน"),
    ("ก็ดีนะ ฉันไม่สำคัญอยู่แล้ว", "งอน"),
    ("เธอก็ไม่เคยสนใจฉันอยู่แล้ว", "งอน"),
    ("ฉันไม่พูดอะไรแล้ว", "งอน"),
    ("ฉันไม่ได้โกรธ แค่ไม่อยากพูดด้วย", "งอน"),
    ("ฉันไม่ได้โกรธ แค่ไม่อยากคุยด้วย", "งอน"),
    ("ฉันไม่พูดอะไรแล้ว เธอทำอะไรก็ทำไป", "งอน"),
    ("ฉันไม่สนใจแล้ว เธอจะทำอะไรก็เชิญ", "งอน"),
    ("ฉันไม่รู้เรื่องอะไรทั้งนั้น", "งอน"),
    ("ฉันไม่แคร์แล้ว", "งอน"),
    ("ฉันไม่ยุ่งแล้ว", "งอน"),
    
    ("วันนี้เธอลืมไปใช่ไหม", "งอน"),
    ("ลืมอะไรเหรอ?", "ปกติ"),
    ("ไม่เป็นไร เธอคงไม่ได้สนใจอยู่แล้ว", "งอน"),
    ("เฮ้ ฉันแค่จำไม่ได้ บอกหน่อยได้ไหม", "ปกติ"),
    
    # ----------------------- ประชด -----------------------
    ("สุดยอดเลยนะ เธอนี่ดีจริงๆ", "ประชด"),
    ("ว้าว เก่งมากเลยนะ", "ประชด"),
    ("ดีจังเลย ฉันชอบแบบนี้มาก", "ประชด"),
    ("โอ้โห ขอบคุณมากเลยนะ", "ประชด"),
    ("ดีใจจังเลย ที่เธอทำแบบนี้", "ประชด"),
    ("สุดยอดเลย ทำได้ดีมาก", "ประชด"),
    ("เธอนี่เป็นคนที่ดีที่สุดเลยนะ", "ประชด"),
    ("เธอวิเศษมากๆ เลย", "ประชด"),
    ("โอ้โห ฉันชอบมากเลยจริงๆ", "ประชด"),
    ("เธอนี่ช่วยฉันได้เยอะมากเลยนะ", "ประชด"),
    ("เก่งจังเลยอ่ะ", "ประชด"),
    ("ดีใจแทนเธอเลยนะ", "ประชด"),
    ("เธอนี่เป็นคนที่ดีมากๆ เลย", "ประชด"),
    
    ("เธอช่วยฉันหน่อยได้ไหม", "ปกติ"),
    ("ไม่ต้องหรอก ฉันรู้ว่าเธอเก่งอยู่แล้ว", "ประชด"),
    ("อย่าพูดแบบนั้นเลย ฉันต้องการความช่วยเหลือจริงๆ", "ปกติ"),
    ("โอ้โห สุดยอดเลยนะ ต้องการฉันแล้วสินะ", "ประชด"),

    # ----------------------- ปกติ -----------------------
    ("วันนี้อากาศดี", "ปกติ"),
    ("พรุ่งนี้ไปเที่ยวกันไหม", "ปกติ"),
    ("ฉันชอบฟังเพลงตอนทำงาน", "ปกติ"),
    ("เธอทำอะไรอยู่", "ปกติ"),
    ("ฉันกินข้าวแล้ว", "ปกติ"),
    ("พรุ่งนี้มีประชุมตอนเช้า", "ปกติ"),
    ("วันนี้ฉันเหนื่อยนิดหน่อย", "ปกติ"),
    ("ฝนตกหนักเลยวันนี้", "ปกติ"),
    ("วันหยุดนี้เธอมีแผนอะไรไหม", "ปกติ"),
    ("ฉันเพิ่งดูหนังเรื่องหนึ่ง สนุกมาก", "ปกติ"),
    ("เธอไปถึงที่นั่นกี่โมง", "ปกติ"),
    ("ฉันกำลังอ่านหนังสือ", "ปกติ"),
    ("เราควรไปกินข้าวด้วยกันบ้าง", "ปกติ"),
    ("วันนี้อากาศดีนะ", "ปกติ"),
    ("ใช่สิ ท้องฟ้าแจ่มใสเลย", "ปกติ"),
    ("ไปเดินเล่นกันไหม", "ปกติ"),
    ("เฮ้ย วันนี้ว่างปะ?", "ปกติ"),
    ("ว่าง ๆ มีไรเหรอ?", "ปกติ"),
    ("ไปกินข้าวกันปะ หิวละ", "ปกติ"),
    ("เอาแบบง่าย ๆ ข้าวมันไก่หน้า ม. ไหม", "ปกติ"),
    ("เออ ดีเลย เดี๋ยวเดินไปเจอที่ร้านเลยปะ", "ปกติ"),
    ("โอเค อีก 10 นาทีเจอกัน", "ปกติ"),
    
    ("สวัสดี", "ปกติ"),
    ("หวัดดี", "ปกติ"),
    ("ไง", "ปกติ"),
    ("โย่ว", "ปกติ"),
    ("เฮ้", "ปกติ"),
    ("อรุณสวัสดิ์", "ปกติ"),
    ("สวัสดีตอนเย็น", "ปกติ"),
    ("สบายดีไหม", "ปกติ"),
    ("เป็นไงบ้าง", "ปกติ"),
    ("ไม่ได้เจอกันนานเลยนะ", "ปกติ"),
    ("ไงเพื่อน", "ปกติ"),
    ("เป็นยังไงบ้างวันนี้", "ปกติ"),
    ("มาทำอะไรตรงนี้", "ปกติ"),
    ("ดีจ้า", "ปกติ"),
    ("โย่ว เป็นไงบ้าง", "ปกติ"),
    ("เฮ้ มีอะไรใหม่ๆบ้าง", "ปกติ"),
    ("อ้าว เจอกันอีกแล้ว", "ปกติ"),
    ("ไง สบายดีมั้ย", "ปกติ"),
    ("สวัสดีค่ะคุณครู", "ปกติ"),
    ("ไง พี่ชาย", "ปกติ"),
    ("ทักครับพี่!", "ปกติ"),
    ("สวัสดีค่ะคุณลูกค้า", "ปกติ"),
    ("เฮ้ น้องชาย วันนี้เรียนเป็นไงบ้าง", "ปกติ"),
    ("ไงพ่อหนุ่ม วันนี้ทำงานหนักไหม", "ปกติ"),
    
    #----------------------------------------------------
    ("เธอไปกับเพื่อนคนนั้นมาอีกแล้วใช่ไหม", "น้อยใจ"),
    ("ก็ใช่ แต่แค่ไปทำงานด้วยกัน", "ปกติ"),
    ("ไม่เป็นไร เธออยากทำอะไรก็ทำไปเถอะ", "งอน"),
    ("อย่าพูดแบบนั้นสิ ฉันไม่ได้คิดอะไร", "ปกติ"),
    ("ว้าว ดีจังเลยนะ เธอนี่สุดยอดจริงๆ", "ประชด"),
    ("เธอกำลังประชดฉันอยู่ใช่ไหม?", "ปกติ"),
    
    ("เธอยังกล้ามาทักฉันอีกเหรอ", "งอน"),
    ("อ๋อ สวัสดี! ไม่คิดว่าจะได้เจอกันอีก", "ประชด"),
    ("เหอะๆ ทักฉันทำไมเหรอ", "น้อยใจ"),
    
    ("อืม วันนี้ว่างเนอะ พอจะจำได้ว่ามีเพื่อนคนนี้อยู่บ้างมั้ย", "ปกติ"),
    ("โห มาแนวนี้เลยเหรอ", "ปกติ"),
    ("ก็เห็นตอบคนอื่นเร็วดี แชทเรานี่คงต้องรอชาติหน้า", "ประชด"),
    ("ไม่เอาน่า อย่าคิดมากเลย แค่ยุ่งจริง ๆ", "ปกติ"),
    ("อ๋อออ ยุ่งกับโลกทั้งใบ ยกเว้นเราคนเดียวอะเนอะ เข้าใจ ๆ", "ประชด"),
    ("โอ้ย ประชดเก่งเกิน ไปเรียนจากไหนมาเนี่ย", "ประชด"),
    ("เรียนจากประสบการณ์ไง คนโดนลืบบ่อย ๆ เขาเป็นกัน", "ประชด"),
    ("โอเค ๆ ยอมละ มื้อนี้เลี้ยงเองเลย หายงอนยัง", "ปกติ"),
    ("หายแล้วมั้ง...ถ้าร้านที่ไปมีชานมอะนะ", "ประชด"),

    ("วันนี้ก็ไม่ทักมาเลยนะ", "ปกติ"),
    ("ขอโทษนะ วันนี้ยุ่งจริง ๆ", "ปกติ"),
    ("ไม่เป็นไร เราไม่ใช่คนสำคัญอยู่แล้ว", "น้อยใจ"),
    ("เราก็แค่รออยู่ตรงนี้แหละ ไม่ได้หวังอะไรหรอก", "น้อยใจ"),
    ("แค่คำว่า 'คิดถึง' บางทีมันก็ดีพอแล้วนะ", "น้อยใจ"),
    ("เราคงคาดหวังมากเกินไปเองแหละ", "น้อยใจ"),
    ("ไม่ได้โกรธนะ แค่น้อยใจนิดหน่อย", "น้อยใจ"),
    ("เดี๋ยวเราก็หายไปเอง จะได้ไม่เป็นภาระใคร", "น้อยใจ"),
    ("เธอคงมีคนอื่นที่สำคัญกว่าอยู่แล้วแหละ", "น้อยใจ"),
    ("อย่าคิดมากเลย เราชินแล้ว", "น้อยใจ"),
    
    ("หวัดดี วันนี้ไปเรียนมั้ย", "ปกติ"),
    ("ไปดิ เจอกันหน้าตึกเดิมนะ", "ปกติ"),
    
    ("เห็นเธอตอบแชทคนอื่นเร็วมากเลยนะ", "ประชด"),
    ("ก็มันเรื่องงานนี่นา", "ปกติ"),
    
    ("ไม่เป็นไรหรอก เราชินแล้ว", "น้อยใจ"),
    ("เฮ้ย อย่าพูดงั้นดิ เราไม่ได้ตั้งใจเมินเลย", "ปกติ"),
    
    ("ไม่ต้องอธิบายก็ได้ เข้าใจอยู่แล้ว", "งอน"),
    ("โอ้ย อย่าเงียบแบบนี้ดิ บอกมาเถอะว่าโกรธอะไร", "ปกติ"),
    
    ("ก็แค่รู้สึกว่าเราไม่สำคัญเท่าคนอื่นแค่นั้นเอง", "น้อยใจ"),
    ("ไม่ใช่แบบนั้นเลยนะ เราแค่วุ่น ๆ ช่วงนี้จริง ๆ", "ปกติ"),
    
    ("อ๋อออ เข้าใจแล้ว วุ่นวายกับคนอื่น แต่เราไม่เกี่ยว", "ประชด"),
    ("เธอนี่ประชดเก่งขึ้นทุกวันเลยนะ", "ประชด"),
    
    ("ไม่ต้องคุยก็ได้มั้ง ถ้ามันลำบากขนาดนั้น", "งอน"),
    ("เดี๋ยว ๆ อย่าเป็นแบบนี้เลยนะ ขอโทษจริง ๆ", "ปกติ"),
    
    ("ขอโทษทีนะ ที่คาดหวังว่าเธอจะสนใจเราบ้าง", "น้อยใจ"),
    ("โอเค เราผิดเอง เดี๋ยววันนี้เลี้ยงชานมเลย", "ปกติ"),
    
    ("ถ้ามีไข่มุกเพิ่มจะหายโกรธเลย", "ประชด"),
    ("เอ้า! ได้เลย ไข่มุกสองเท่าก็ยอม", "ปกติ")
]
labels = {"น้อยใจ": 0, "งอน": 1, "ประชด": 2, "ปกติ": 3}
# -------------------------------------------

MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# เตรียมข้อมูล
texts = [d[0] for d in data]
label_names = [d[1] for d in data]
label_ids = [labels[name] for name in label_names]

# 1. แบ่งข้อมูลเป็น Training และ Validation Sets (เช่น 80% train, 20% validation)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, label_ids, test_size=0.2, random_state=42, stratify=label_ids # stratify ช่วยให้สัดส่วน label ใกล้เคียงกัน
)

# 2. Tokenize แยกกันสำหรับ Train และ Validation
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")
val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors="pt")

train_labels_tensor = torch.tensor(train_labels)
val_labels_tensor = torch.tensor(val_labels)

# 3. สร้าง Dataset และ DataLoader แยกกัน
class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Ensure all tensor values are retrieved for the specific index
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx].clone().detach()
        return item

train_dataset = EmotionDataset(train_encodings, train_labels_tensor)
val_dataset = EmotionDataset(val_encodings, val_labels_tensor)

# ปรับ Batch Size ตามความเหมาะสมและหน่วยความจำที่มี
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8) # Validation ไม่ต้อง shuffle

# 4. ปรับโมเดลใน PyTorch Lightning ให้มี validation_step
class EmotionClassifier(pl.LightningModule):
    def __init__(self, model_name, learning_rate=5e-5): # เพิ่ม learning_rate เป็น argument
        super().__init__()
        self.save_hyperparameters() # บันทึก hyperparameters เช่น learning_rate
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(labels) # ใช้ len(labels) เพื่อความยืดหยุ่น
        )

    def forward(self, input_ids, attention_mask, labels=None):
         # Pass labels to the model if available (during training/validation)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(**batch) # Pass the whole batch dictionary
        loss = outputs.loss
        self.log('train_loss', loss) # Log training loss
        return loss

    # เพิ่ม validation_step
    def validation_step(self, batch, batch_idx):
        outputs = self(**batch) # Pass the whole batch dictionary
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        # คำนวณ accuracy (ตัวอย่าง)
        acc = (preds == batch['labels']).float().mean()
        self.log('val_loss', loss, prog_bar=True) # Log validation loss
        self.log('val_acc', acc, prog_bar=True)  # Log validation accuracy
        return loss # หรือ return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        # ใช้ learning_rate ที่รับเข้ามา
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

# สร้างโมเดลและ Trainer
model = EmotionClassifier(MODEL_NAME, learning_rate=5e-5) # สามารถเปลี่ยน learning rate ตรงนี้ได้ 1e-5, 3e-5, 1e-4 5e-5

# เพิ่ม Callback สำหรับ Early Stopping (ตัวอย่าง)
from pytorch_lightning.callbacks import EarlyStopping

early_stop_callback = EarlyStopping(
   monitor='val_loss', # หรือ 'val_acc'
   patience=3,         # จำนวน epochs ที่จะรอถ้า val_loss ไม่ลดลง
   verbose=True,
   mode='min'          # 'min' สำหรับ loss, 'max' สำหรับ accuracy
)

# เทรนโมเดล โดยใส่ val_dataloader และ callbacks เข้าไปด้วย
trainer = pl.Trainer(
    max_epochs=10, # เพิ่มจำนวน epochs ได้ เพราะมี early stopping ช่วย
    accelerator="auto",
    callbacks=[early_stop_callback] # เพิ่ม callback
)
trainer.fit(model, train_dataloader, val_dataloader) # ใส่ val_dataloader เข้าไป

# --- (ส่วนการ Save และ Predict เหมือนเดิม แต่ควรใช้โมเดลที่ดีที่สุดจากการเทรน) ---

# โหลดโมเดลที่ดีที่สุดที่ถูกบันทึกโดย Trainer (ถ้ามีการตั้งค่า checkpoint callback)
# หรือใช้โมเดลสุดท้ายหลังจากการเทรน
# model = EmotionClassifier.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
# model.eval() # ตั้งเป็น evaluation mode

# บันทึกโมเดลสุดท้ายหลังเทรนเสร็จ
model_save_path = 'saved_model_best' # ตั้งชื่อ path ใหม่
model.model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

# ฟังก์ชัน Predict (ปรับปรุงเล็กน้อยให้ใช้ model ที่โหลดมา หรือ model สุดท้าย)
loaded_tokenizer = AutoTokenizer.from_pretrained(model_save_path)
loaded_model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
# ถ้าไม่ได้ใช้ PyTorch Lightning ต่อ หรือต้องการใช้แค่ model ของ Hugging Face
# ให้สร้าง instance ของ EmotionClassifier ใหม่ แล้วโหลด state_dict หรือใช้ loaded_model โดยตรง

def predict_hf(text, model_hf, tokenizer_hf):
    inputs = tokenizer_hf(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad(): # ไม่ต้องคำนวณ gradient ตอน predict
        outputs = model_hf(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # สร้าง label map แบบกลับด้านเพื่อแสดงผล
    id2label = {v: k for k, v in labels.items()}
    # แสดงผลลัพธ์
    results = {id2label[i]: round(prob.item() * 100, 2) for i, prob in enumerate(probs[0])}
    # เรียงตาม probability
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    return sorted_results


print("--- Prediction ---")
print(predict_hf("เออ ดีมากเลยนะ", loaded_model, loaded_tokenizer))
print()
print(predict_hf("สวัสดีครับ", loaded_model, loaded_tokenizer))
print()
# ข้อความยาวๆ อาจจะถูกตัด (truncate) หรือต้องจัดการ padding ให้เหมาะสม
long_text = "ทำไมเธอถึงไม่ตอบแชทฉันเลย ขอโทษนะ ฉันยุ่งอยู่จริงๆ อ๋อ งั้นก็ไม่เป็นไร ฉันคงไม่สำคัญอยู่แล้ว ไม่ใช่แบบนั้นนะ! วันนี้เธอลืมไปใช่ไหม ลืมอะไรเหรอ?"
print(predict_hf(long_text, loaded_model, loaded_tokenizer))
