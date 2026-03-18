import torch
import numpy as np
from typing import Dict


def flatten_state(state: Dict, device: str) -> torch.Tensor:

    vec = np.concatenate([
        state["tcp_pose"],
        state["tcp_vel"],
        state["gripper_pos"],
        state["gripper_vec"],
    ])

    return torch.from_numpy(vec).float().to(device=device).unsqueeze(0)


def image_to_tensor(image: np.ndarray, device: str, size: int = None) -> torch.Tensor:
    tensor = torch.from_numpy(image.copy()).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    if size is not None:
        tensor = torch.nn.functional.interpolate(
            tensor, size=(size, size), mode="bilinear", align_corners=False
        )
    return tensor.to(device=device)
