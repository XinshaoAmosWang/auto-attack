import os

import rootpath
import torch

from models.resnet import ResNet18

REPO_PATH = rootpath.detect()

def load_resnet18(device: str = "cuda:0"):
    model_path = os.path.join(
        REPO_PATH,
        "models",
        "model_test.pt"
    )

    # load model
    model = ResNet18()
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt)

    # use cuda and eval modes
    model.to(device=device)
    model.eval()

    return model