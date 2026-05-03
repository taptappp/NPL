import pandas as pd
from sklearn.model_selection import train_test_split
import os

INPUT = "data/processed/train_clean.csv"
OUT_DIR = "data/split"

df = pd.read_csv(INPUT)

train, temp = train_test_split(
    df, test_size=0.2, stratify=df["label_id"], random_state=42
)
val, test = train_test_split(
    temp, test_size=0.5, stratify=temp["label_id"], random_state=42
)

os.makedirs(OUT_DIR, exist_ok=True)
train.to_csv(f"{OUT_DIR}/train.csv", index=False, encoding="utf-8-sig")
val.to_csv(f"{OUT_DIR}/val.csv", index=False, encoding="utf-8-sig")
test.to_csv(f"{OUT_DIR}/test.csv", index=False, encoding="utf-8-sig")

print(" Đã chia train / val / test")
