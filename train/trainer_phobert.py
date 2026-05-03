import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, device, loss_fn=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        progress_bar = tqdm(dataloader, desc="Training", leave=False)

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits

            if self.loss_fn:
                loss = self.loss_fn(logits, labels)
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / max(len(dataloader), 1)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')

        return avg_loss, acc, f1

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits

                if self.loss_fn:
                    loss = self.loss_fn(logits, labels)
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss

                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(len(dataloader), 1)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')

        return avg_loss, acc, f1