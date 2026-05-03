import pandas as pd
import os

INPUT = "data/processed/train_auto.csv"
OUTPUT = "data/processed/train_clean.csv"
THRESHOLD = 0.4

df = pd.read_csv(INPUT)
df = df[df["confidence"] >= THRESHOLD]

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print(f"Giữ lại {len(df)} mẫu chất lượng cao")
