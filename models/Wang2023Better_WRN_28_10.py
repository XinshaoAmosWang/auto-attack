import os

import rootpath
import torch
from torch import nn

from robustbench.model_zoo.architectures.dm_wide_resnet import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    DMWideResNet,
)

REPO_PATH = rootpath.detect()


def load_Wang2023Better_WRN_28_10(device: str = "cuda:0"):
    model_path = os.path.join(
        REPO_PATH,
        "models",
        "cifar10",
        "Linf",
        "Wang2023Better_WRN-28-10.pt",
    )
    # load state
    checkpoint = torch.load(model_path, map_location=torch.device(device))

    # load model
    model = DMWideResNet(
        num_classes=10,
        depth=28,
        width=10,
        activation_fn=nn.SiLU,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
    )
    model.load_state_dict(checkpoint, strict=True)

    # use cuda and eval modes
    model.to(device=device)
    model.eval()

    return model
