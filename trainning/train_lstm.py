import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train.trainer_lstm import LSTMTrainer

if __name__ == "__main__":
    trainer = LSTMTrainer(
        model_dir="saved_models/lstm"
    )

    trainer.train("data/split/train.csv")