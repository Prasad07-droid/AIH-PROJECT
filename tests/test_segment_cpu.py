"""CPU-only inference regression tests."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from woundtrack.src.segment import predict_mask


class ConstantMaskModel(nn.Module):
    """Tiny deterministic stand-in that records the input device."""

    def __init__(self) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(4.0))
        self.input_devices: list[str] = []

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.input_devices.append(image.device.type)
        return self.logit.expand(image.shape[0], 1, image.shape[2], image.shape[3])


def test_predict_mask_forces_passed_model_and_tensor_to_cpu() -> None:
    model = ConstantMaskModel()
    image = np.zeros((23, 35, 3), dtype=np.uint8)

    # Even an accelerator request must be ignored for inference.
    mask = predict_mask(image, model=model, device="cuda")

    assert model.logit.device.type == "cpu"
    assert model.input_devices == ["cpu"]
    assert mask.shape == image.shape[:2]
    assert mask.dtype == np.uint8
    assert np.all(mask == 1)
