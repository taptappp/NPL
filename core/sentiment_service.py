import torch
import json
from transformers import AutoTokenizer, RobertaForSequenceClassification

MODEL_PATH = "saved_models/phobert/model.pt"
MODEL_NAME = "vinai/phobert-base"
LABEL_PATH = "config/labels_config.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load label map
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)

id2label = {int(k): v for k, v in label_map.items()}

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# Load model
model = RobertaForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_map)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


def predict_sentiment(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()

    return {
        "text": text,
        "label_id": pred,
        "label": id2label[pred]
    }