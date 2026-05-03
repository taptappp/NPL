import torch
from torch.utils.data import Dataset
import pandas as pd

class SentimentDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        tokenizer,
        label_map: dict = None,
        max_len: int = 128,
        is_train: bool = True
    ):
        """
        Args:
            csv_path: Đường dẫn file CSV dữ liệu.
            tokenizer: Tokenizer của PhoBERT (AutoTokenizer).
            label_map: Dict map từ nhãn chữ sang số. VD: {'tiêu_cực': 0, 'tích_cực': 1}.
                       Nếu None, mặc định cột label trong CSV đã là số int.
            max_len: Độ dài tối đa của câu (padding/truncation).
            is_train: Nếu True, sẽ đọc cột 'label'. Nếu False (lúc dự đoán), bỏ qua label.
        """
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train
        self.label_map = label_map
        if "clean_text" in self.df.columns:
            self.text_col = "clean_text"
        elif "text" in self.df.columns:
            self.text_col = "text"
        else:
            raise ValueError(f"File CSV {csv_path} phải có cột 'clean_text' hoặc 'text'")

        if self.is_train:
            if "label" not in self.df.columns:
                raise ValueError(f"File CSV {csv_path} dùng để train phải có cột 'label'")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        text = str(row[self.text_col])

        # Tokenize (Chuyển chữ thành số ID của PhoBERT)
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if self.is_train:
            raw_label = row["label"]
            
            if self.label_map is not None:
  
                label_id = self.label_map.get(raw_label, -1)
                if label_id == -1:
                    print(f" Cảnh báo: Không tìm thấy nhãn '{raw_label}' trong label_map tại dòng {idx}.")
            else:
                label_id = int(raw_label)

            item["labels"] = torch.tensor(label_id, dtype=torch.long)

        return item