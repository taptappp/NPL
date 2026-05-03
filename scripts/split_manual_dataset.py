import pandas as pd
import os
from sklearn.model_selection import train_test_split

INPUT_FILE = "data/processed/manual_all.csv"
OUT_DIR = "data/split/manual"

os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(INPUT_FILE):
    print(f" Không tìm thấy file {INPUT_FILE}. Hãy chạy build_manual_dataset.py trước!")
    exit()

df = pd.read_csv(INPUT_FILE)
print(f" Đã đọc {len(df)} dòng dữ liệu.")

train_parts, val_parts, test_parts = [], [], []


for label_id in sorted(df["label"].unique()):
    df_label = df[df["label"] == label_id]
    

    if len(df_label) < 3:
        print(f" Nhãn {label_id} chỉ có {len(df_label)} dòng -> Dồn hết vào Train.")
        train_parts.append(df_label)
        continue

    # Chia Train (70%) - Temp (30%)
    train, temp = train_test_split(
        df_label,
        test_size=0.3,
        random_state=42,
        shuffle=True
    )

    # Chia Temp -> Val (15%) - Test (15%)
    val, test = train_test_split(
        temp,
        test_size=0.5,
        random_state=42,
        shuffle=True
    )

    train_parts.append(train)
    val_parts.append(val)
    test_parts.append(test)

if not train_parts:
    print("Lỗi: Không có dữ liệu nào được chia!")
    exit()

train_df = pd.concat(train_parts).sample(frac=1, random_state=42)
val_df   = pd.concat(val_parts).sample(frac=1, random_state=42)
test_df  = pd.concat(test_parts).sample(frac=1, random_state=42)

# Lưu file
train_df.to_csv(f"{OUT_DIR}/train.csv", index=False, encoding="utf-8-sig")
val_df.to_csv(f"{OUT_DIR}/val.csv", index=False, encoding="utf-8-sig")
test_df.to_csv(f"{OUT_DIR}/test.csv", index=False, encoding="utf-8-sig")

print("\n Chia dataset MANUAL hoàn tất!")
print(f"   - Train: {train_df.shape[0]} dòng")
print(f"   - Val:   {val_df.shape[0]} dòng")
print(f"   - Test:  {test_df.shape[0]} dòng")