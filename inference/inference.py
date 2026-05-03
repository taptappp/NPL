import json
import torch
import os
from transformers import AutoTokenizer
from models.transformer.phobert_model import PhoBERTClassifier

try:
    from underthesea import word_tokenize
except ImportError:
    print("CẢNH BÁO: Chưa cài underthesea. Kết quả dự đoán có thể không chính xác.")
    print(" Hãy chạy: pip install underthesea")
    def word_tokenize(text, format=None): return text

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_PATH = "config/labels.json"
MODEL_PATH = "saved_models/phobert_best.pt" 
BASE_MODEL = "vinai/phobert-base"


if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)
        id2label = {int(v): k for k, v in label_map.items()}
else:
    print(f" Không tìm thấy {CONFIG_PATH}")
    id2label = {}

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model = PhoBERTClassifier(num_labels=len(id2label))

if os.path.exists(MODEL_PATH):
    print(f" Đang load model từ {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
else:
    print(f" Không tìm thấy file weights tại {MODEL_PATH}. Model sẽ dự đoán ngẫu nhiên!")

model.to(DEVICE)
model.eval()

def predict(text: str, threshold=0.0):
    """
    Hàm dự đoán cảm xúc cho một câu văn bản.
    """

    segmented_text = word_tokenize(text, format="text")


    with torch.no_grad():
        inputs = tokenizer(
            segmented_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        outputs = model(input_ids, attention_mask)      

        probs = torch.softmax(outputs["logits"], dim=-1)[0]
        confidence, pred_id = torch.max(probs, dim=-1)
        
        pred_id = pred_id.item()
        confidence = confidence.item()

    label_text = id2label.get(pred_id, "Unknown")
    
    note = ""
    if confidence < threshold:
        note = "(Độ tin cậy thấp)"

    return {
        "text": text,
        "segmented": segmented_text, 
        "label": label_text,
        "confidence": round(confidence, 4),
        "note": note
    }
if __name__ == "__main__":
    print(f" Hệ thống đã sẵn sàng trên thiết bị: {DEVICE}")
    print("Gõ 'thoát' để dừng chương trình.")
    
    while True:
        text = input("\n Nhập câu tiếng Việt: ")
        if text.lower() in ["thoát", "exit", "quit"]:
            break
        
        if not text.strip():
            continue

        result = predict(text)
        
        print("-" * 30)
        print(f"Cảm xúc: {result['label'].upper()}")
        print(f"Độ tin cậy: {result['confidence']*100:.2f}% {result['note']}")
        print(f"Debug (Tách từ): {result['segmented']}")
        print("-" * 30)