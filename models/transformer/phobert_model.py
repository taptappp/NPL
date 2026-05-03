import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Optional # Để tương thích tốt hơn với các phiên bản Python cũ

class PhoBERTClassifier(nn.Module):
    def __init__(self, num_labels: int = 7, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_labels = num_labels

        # Load PhoBERT bare model
        self.phobert = AutoModel.from_pretrained(
            "vinai/phobert-base",
            return_dict=True
        )

        hidden_size = self.phobert.config.hidden_size
        
        # Dropout giúp giảm overfitting (0.1 - 0.3 là chuẩn)
        self.dropout = nn.Dropout(0.3)
        
        # Lớp phân loại cuối cùng
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Xử lý Loss function
        if class_weights is not None:
            # QUAN TRỌNG: Đảm bảo weights cũng phải được đưa vào đúng thiết bị (CPU/GPU) cùng với model
            # Dòng này sửa lỗi tiềm ẩn: RuntimeError: Expected device cuda:0 but got device cpu
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ):
        # 1. Chạy qua PhoBERT
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 2. Lấy vector đặc trưng của token đầu tiên (<s> - vị trí 0)
        # Vector này đại diện cho ý nghĩa cả câu
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # 3. Qua Dropout và Classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # 4. Tính Loss (nếu đang train)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {
            "loss": loss,
            "logits": logits
        }

    @torch.no_grad()
    def predict(self, input_ids, attention_mask):
        self.eval()
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Trả về nhãn có xác suất cao nhất
        return torch.argmax(outputs["logits"], dim=-1)