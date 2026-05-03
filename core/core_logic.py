import torch
import json
import os
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, RobertaForSequenceClassification
from .utils import normalize_vietnamese 

# --- CẤU HÌNH ---
MODEL_PATH = "saved_models/phobert/model.pt"
CONFIG_PATHS = ["config/label_config.json", "config/label.json"] 
BASE_MODEL = "vinai/phobert-base"
MAX_LEN = 128

class SentimentService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"AI Core loading on: {self.device}")
        
        # Load Config Labels
        self.id2label = {}
        found_config = False
        
        for path in CONFIG_PATHS:
            if os.path.exists(path):
                print(f"Đã tìm thấy config tại: {path}")
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                        # chuyen string -> int
                        self.id2label = {int(k): v for k, v in config.items()}
                        found_config = True
                        break
                except Exception as e:
                    print(f"⚠️ Lỗi đọc file {path}: {e}")

        # mặc định 7 nhãn nếu không có file
        if not found_config:
            print(" Không tìm thấy file config, sử dụng cấu hình mặc định 7 nhãn.")
            self.id2label = {
                0: "trung_tính",
                1: "vui",
                2: "buồn",
                3: "sợ_hãi",
                4: "tức_giận",
                5: "ghê_tởm",
                6: "ngạc_nhiên"
            }

        num_labels = len(self.id2label)
        print(f"Số lượng nhãn nhận diện: {num_labels}")

        # 2. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)

        # 3. Load Model Structure
        # Khởi tạo khung model với đúng số lượng nhãn (7)
        config = AutoConfig.from_pretrained(BASE_MODEL, num_labels=num_labels)
        self.model = RobertaForSequenceClassification.from_pretrained(BASE_MODEL, config=config)

        # 4. Load Trained Weights
        if os.path.exists(MODEL_PATH):
            print(f" Đang load weights từ {MODEL_PATH}...")
            state_dict = torch.load(MODEL_PATH, map_location=self.device)
            
            # Load vào model
            try:
                self.model.load_state_dict(state_dict)
                print("Đã load model thành công!")
            except RuntimeError as e:
                print(f" LỖI SIZE : Model code đang có {num_labels} nhãn, nhưng file weights khác.")
                print("Chi tiết lỗi:", e)
                exit()
        else:
            print(f" LỖI: Không tìm thấy file model tại {MODEL_PATH}")
            print("Hãy chắc chắn bạn đã copy file phobert_best.pt từ Colab về thư mục saved_models")

        self.model.to(self.device)
        self.model.eval()

    def predict_sentiment(self, text):
        clean_text = normalize_vietnamese(text)
        if not clean_text:
            return {"status": "error", "msg": "Text rỗng"}

        inputs = self.tokenizer(
            clean_text,
            return_tensors="pt",
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length"
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            
            max_prob, pred_id = torch.max(probs, dim=1)
            pred_id = pred_id.item()
            confidence = max_prob.item()

        label_name = self.id2label.get(pred_id, f"Unknown ({pred_id})")

        return {
            "text": text,
            "prediction": label_name,
            "confidence": round(confidence, 4),
            "probabilities": {self.id2label.get(i, str(i)): round(p.item(), 4) for i, p in enumerate(probs[0])}
        }

sentiment_service = SentimentService()