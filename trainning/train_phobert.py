import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, RobertaForSequenceClassification
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import os
import json
import random
import sys


current_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(current_path))
if root_path not in sys.path:
    sys.path.append(root_path)

from train.trainer_phobert import Trainer


MODEL_NAME = "vinai/phobert-base"
TRAIN_PATH = "data/split/train.csv"
VAL_PATH   = "data/split/val.csv"
SAVE_PATH  = "saved_models/phobert/model.pt"
LABEL_PATH = "config/labels_config.json"

MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 10
LR = 2e-5
SEED = 42
PATIENCE = 3


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PhoBERTDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df["phobert_text"].values
        self.labels = df["label"].values
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long)
        }


def load_data(path):
    if not os.path.exists(path):
        print(f" Không tìm thấy: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = df.dropna(subset=["phobert_text", "label"])
    df["label"] = df["label"].astype(int)

    print(f" Loaded {len(df)} samples: {path}")
    print(" Label distribution:\n", df["label"].value_counts().sort_index())

    return df


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Device: {device}")

  
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    label2id = label_map
    id2label = {v: k for k, v in label_map.items()}
    num_labels = len(label_map)

    print(" Labels:", id2label)


    train_df = load_data(TRAIN_PATH)
    val_df   = load_data(VAL_PATH)

    if train_df.empty:
        print(" Train data rỗng!")
        return

  
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    train_loader = DataLoader(
        PhoBERTDataset(train_df, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        PhoBERTDataset(val_df, tokenizer),
        batch_size=BATCH_SIZE
    )


    y_train = train_df["label"].values

    class_counts = np.bincount(y_train, minlength=num_labels)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_labels

    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    print("Class counts:", class_counts)
    print(" Class weights:", class_weights)


    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    trainer = Trainer(model, optimizer, device, loss_fn)


    best_f1 = 0
    no_improve = 0

    print("\nSTART TRAINING\n")

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")

        train_loss, train_acc, train_f1 = trainer.train_epoch(train_loader)
        val_loss, val_acc, val_f1 = trainer.evaluate(val_loader)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0

            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)

            print(f" Saved best model (F1={best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(" Early stopping")
                break

    print("\n FINAL REPORT")

    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            outputs = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device)
            )
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    print(classification_report(all_labels, all_preds))


if __name__ == "__main__":
    main()