from fastai.vision import F

import torch

from pathlib import Path

__all__ = ["BilinearWrapper"]


class BilinearWrapper(torch.nn.Module):
    def __init__(self, model, scale=4, mode="bilinear"):
        super().__init__()
        self.model = model
        self.scale = scale
        self.mode = mode
        # self.mean = torch.nn.Parameter(torch.tensor([0.4850, 0.4560, 0.4060]))
        # self.std = torch.nn.Parameter(torch.tensor([0.2290, 0.2240, 0.2250]))
        self.mean = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
        self.std = torch.nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

    def forward(self, x):
        x = (x - self.mean[None, ..., None, None]) / self.std[None, ..., None, None]
        pred = self.model(
            F.interpolate(
                x, scale_factor=self.scale, mode=self.mode, align_corners=False
            )
        )
        pred = pred * self.std[None, ..., None, None] + self.mean[None, ..., None, None]
        return torch.mean(pred, dim=1, keepdim=True)


class Learner(torch.nn.Module):
    def __init__(self, learner: str):
        super().__init__()
        self.interp = BilinearWrapper(
            torch.load(Path(__file__).parent / learner).float()
        )

    def forward(self, x):
        return self.interp(x)
