import json
import torch
import os
from transformers import AutoTokenizer
from core.engine import SentimentEngine
from models.transformer.phobert_model import PhoBERTClassifier

try:
    from underthesea import word_tokenize
except ImportError:
    def word_tokenize(text, format=None): return text

class HybridPredictor:
    def __init__(self, model_path="saved_models/phobert_classifier.pt", config_dir="config"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conf_rule = 0.6
        self.conf_model = 0.75 

        label_path = os.path.join(config_dir, "labels.json")
        with open(label_path, "r", encoding="utf-8") as f:
            self.id2label = {int(k): v for k, v in json.load(f).items()}
        

        self.rule_engine = SentimentEngine(config_dir=config_dir)

        print(f"Loading PhoBERT model to {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        
        self.model = PhoBERTClassifier(num_labels=len(self.id2label))
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(f"⚠️ CẢNH BÁO: Không tìm thấy file weights tại {model_path}. Model sẽ output ngẫu nhiên.")
        
        self.model.to(self.device)
        self.model.eval()
        print("✅ HybridPredictor Ready!")

    def _preprocess_for_phobert(self, text: str):
        """
        PhoBERT cần input dạng: "Tôi đang đi_học" thay vì "Tôi đang đi học"
        """
        return word_tokenize(text, format="text")

    def predict(self, text: str):
        rule_label, rule_scores = self.rule_engine.analyze_paragraph_rules(text)
        
        rule_conf = 0.0
        rule_label_norm = rule_label.lower().replace(" ", "_") if rule_label else "trung_tính"

        if rule_scores:
            rule_conf = max(rule_scores.values()) / 100.0

        model_conf = 0.0
        model_label = "unknown"
        
        segmented_text = self._preprocess_for_phobert(text)

        with torch.no_grad():
            inputs = self.tokenizer(
                segmented_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            outputs = self.model(input_ids, attention_mask)
            
            probs = torch.softmax(outputs["logits"], dim=-1)[0]
            conf, pred_id = torch.max(probs, dim=-1)
            
            model_conf = conf.item()
            model_label = self.id2label.get(pred_id.item(), "unknown")

        
        final_label = model_label
        source = "model"

        if rule_conf >= self.conf_rule:
            final_label = rule_label_norm
            source = "rule_high_confidence"
        elif model_conf < 0.5:
            source = "low_confidence_both"
        
        return {
            "input": text,
            "final_prediction": final_label,
            "source": source,
            "details": {
                "rule": {"label": rule_label_norm, "conf": round(rule_conf, 3)},
                "model": {"label": model_label, "conf": round(model_conf, 3)}
            }
        }

hybrid_predictor = HybridPredictor()