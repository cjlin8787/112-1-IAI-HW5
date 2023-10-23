import numpy as np
import torch


class HW5Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.setup_model()

    def setup_model(self) -> None:
        """Setup model architecture here"""
        # TODO, change this
        return

    def train(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Train the model"""
        # TODO, change this
        for batch in dataloader:
            pass
        return

    def predict(self, dataloader: torch.utils.data.DataLoader) -> np.array:
        """Predict the output based on trained model"""
        # TODO, change this
        pred = np.zeros(1)
        for batch in dataloader:
            pass
        return pred
