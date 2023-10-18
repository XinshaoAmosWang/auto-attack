import os

import rootpath
import torch

from models.resnet18 import load_resnet18
from models.resnet import ResNet18

REPO_PATH = rootpath.detect()

def test_model_1():
    net = ResNet18()
    # Note: the output is logits
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


def test_model_2():
    # config the device
    device = "cuda:0"

    model = load_resnet18(device=device)

    # try forward processing
    # Note: the output is logits
    y = model(torch.randn(1, 3, 32, 32).to(device=device))
    print(y.size())