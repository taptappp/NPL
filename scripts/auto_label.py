import sys
import os
import pandas as pd
# Thêm thư viện tách từ
try:
    from underthesea import word_tokenize
except ImportError:
    def word_tokenize(text, format=None): return text
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path: sys.path.insert(0, ROOT_DIR)
from core.engine import SentimentEngine

INPUT_FILE = "data/raw/nhacbuon.csv"
OUTPUT_FILE = "data/processed/train_auto.csv"
LABEL_CONFIG_PATH = "config/labels.json" 

def auto_label():
    engine = SentimentEngine()
    

    import json
    with open(LABEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
        LABEL_MAP = json.load(f) 

    df = pd.read_csv(INPUT_FILE)
    

    possible_cols = ["clean_text", "comment", "text", "content"]
    text_col = next((c for c in possible_cols if c in df.columns), None)
    if not text_col: raise ValueError("Không tìm thấy cột nội dung text")

    results = []
    print("⏳ Đang gán nhãn tự động...")
    
    for _, row in df.iterrows():
        raw_text = str(row[text_col]) if not pd.isna(row[text_col]) else ""
        if not raw_text.strip(): continue


        label_text, scores = engine.analyze_paragraph_rules(raw_text)
        

        total = sum(scores.values()) if scores else 0
        confidence = max(scores.values()) / total if total > 0 else 0.0


        segmented_text = word_tokenize(raw_text, format="text")

        label_key = label_text.lower() 
        label_id = LABEL_MAP.get(label_key, 0) 

        results.append({
            "text": raw_text,        
            "clean_text": segmented_text, 
            "label": label_id,      
            "label_name": label_key,  
            "confidence": round(confidence, 3),
            "source": "auto_rule"
        })

    out_df = pd.DataFrame(results)
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Hoàn tất! Đã lưu {len(out_df)} dòng vào {OUTPUT_FILE}")

if __name__ == "__main__":
    auto_label()