import torch
from torch import nn

from data.cifar10 import load_cifar10_test_x_y
from robustbench.model_zoo.architectures.dm_wide_resnet import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    DMWideResNet,
)

ExpectedOutput = torch.Tensor(
    [
        [
            -0.4918,
            -1.1725,
            -0.0515,
            2.1848,
            0.1238,
            1.2371,
            0.3112,
            -0.5574,
            -0.5971,
            -0.9859,
        ]
    ]
)


def test_robustbench_load_model():
    # config the device
    device = "cuda:0"

    # #######################################
    # load data
    x_test, y_test = load_cifar10_test_x_y()
    # #######################################

    # #######################################
    # load model:
    model_path = "models/cifar10/Linf/Wang2023Better_WRN-70-16.pt"
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model = DMWideResNet(
        num_classes=10,
        depth=70,
        width=16,
        activation_fn=nn.SiLU,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
    )
    model.load_state_dict(checkpoint, strict=True)
    model.to(device=device)
    model.eval()
    # #######################################
    # #######################################
    # try forward processing
    # Note: the output is logits
    y = model(x_test[0:1].to(device=device)).detach().cpu()
    assert torch.allclose(y, ExpectedOutput, atol=1e-3)
    # ########################################
