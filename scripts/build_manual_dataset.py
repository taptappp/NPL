import pandas as pd
import os
import json
import re
from sklearn.model_selection import train_test_split

try:
    from underthesea import word_tokenize, text_normalize
except ImportError:
    print(" Chưa cài underthesea. pip install underthesea")
    def word_tokenize(text, format=None): return text
    def text_normalize(text): return text

INPUT_DIR = "data/raw/manual"
SPLIT_DIR = "data/split"
CONFIG_PATH = "config/labels_config.json"
PROCESSED_FILE = "data/processed/manual_all.csv"

def clean_text_base(text, keep_caps=False):
    if pd.isna(text): return ""
    text = str(text)

    text = text.replace("\n", " ")
    text = text_normalize(text)

    if not keep_caps:
        text = text.lower()

    slang_dict = {
        "khum": "không", "hok": "không", "ko": "không", "k": "không",
        "bít": "biết", "cx": "cũng", "đc": "được"
    }
    for k, v in slang_dict.items():
        text = re.sub(rf"\b{k}\b", v, text)

    text = re.sub(r"(.)\1{2,}", r"\1", text)

    emoji_dict = {
        "😊": " vui ", "😁": " vui ", "😄": " vui ", "😆": " vui ",
        "🥰": " vui ", "😍": " vui ", "😋": " vui ", "🤩": " vui ",
        "😭": " buồn ", "😢": " buồn ", "😞": " buồn ", "😔": " buồn ",
        "😥": " buồn ", "🥺": " buồn ",
        "😡": " tức_giận ", "😠": " tức_giận ", "🤬": " tức_giận ",
        "😱": " sợ_hãi ", "😨": " sợ_hãi ", "😰": " sợ_hãi ",
        "🤢": " ghê_tởm ", "🤮": " ghê_tởm ",
        "😲": " ngạc_nhiên ", "😮": " ngạc_nhiên ", "😳": " ngạc_nhiên ",
        "😐": " trung_tính ", "😶": " trung_tính "
    }

    for e, val in emoji_dict.items():
        text = text.replace(e, val)

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s\.\!\?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def handle_negation(text):
    patterns = [
        ("không vui", "không_vui"),
        ("không buồn", "không_buồn"),
        ("không thích", "không_thích"),
        ("chẳng vui", "chẳng_vui"),
        ("chả vui", "chả_vui"),
    ]
    for p, r in patterns:
        text = text.replace(p, r)
    return text


def preprocess_for_nb(text):
    text = clean_text_base(text)
    text = handle_negation(text)
    text = word_tokenize(text, format="text")
    return text


def preprocess_for_lstm(text):
    return clean_text_base(text)

def preprocess_for_phobert(text):
    return clean_text_base(text, keep_caps=True)


def build_dataset():
    if not os.path.exists(CONFIG_PATH):
        print("Không tìm thấy labels_config.json")
        return

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        LABEL_MAP = json.load(f)

    ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

    all_rows = []

    print(" Building dataset...")

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]
    if not files:
        print(" Không có file CSV")
        return

    for filename in files:
        path = os.path.join(INPUT_DIR, filename)

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f" Lỗi đọc {filename}: {e}")
            continue

        if "text" not in df.columns or "label_id" not in df.columns:
            print(f" Bỏ {filename}: thiếu text hoặc label_id")
            continue

        print(f"✔ Đang xử lý {filename} ({len(df)} dòng)")

        for _, row in df.iterrows():
            raw = row["text"]
            if pd.isna(raw): continue

            try:
                f_id = int(float(row["label_id"]))
            except:
                continue

            if f_id not in range(7):
                continue

            raw = str(raw)

            all_rows.append({
                "raw_text": raw,
                "nb_text": preprocess_for_nb(raw),
                "lstm_text": preprocess_for_lstm(raw),
                "phobert_text": preprocess_for_phobert(raw),
                "label": f_id,
                "label_name": ID2LABEL.get(f_id, "unknown"),
                "length": len(raw.split()),
                "num_exclaim": raw.count("!"),
                "num_question": raw.count("?")
            })

    if not all_rows:
        print(" Không có dữ liệu hợp lệ")
        return

    df = pd.DataFrame(all_rows)

    df = df.drop_duplicates(subset=["raw_text"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f" Total samples: {len(df)}")

    stratify = df["label"] if df["label"].value_counts().min() >= 2 else None

    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=stratify
    )

    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42,
        stratify=temp_df["label"] if stratify is not None else None
    )

    os.makedirs(SPLIT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)

    df.to_csv(PROCESSED_FILE, index=False, encoding="utf-8-sig")
    train_df.to_csv(f"{SPLIT_DIR}/train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(f"{SPLIT_DIR}/val.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(f"{SPLIT_DIR}/test.csv", index=False, encoding="utf-8-sig")

    print("DONE")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


if __name__ == "__main__":
    build_dataset()