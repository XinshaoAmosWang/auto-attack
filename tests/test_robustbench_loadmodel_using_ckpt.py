import torch

from data.cifar10 import load_cifar10_test_x_y
from models.Wang2023Better_WRN_70_16 import load_Wang2023Better_WRN_70_16

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
    model = load_Wang2023Better_WRN_70_16(device=device)
    # #######################################
    # #######################################
    # try forward processing
    # Note: the output is logits
    y = model(x_test[0:1].to(device=device)).detach().cpu()
    assert torch.allclose(y, ExpectedOutput, atol=1e-3)
    # ########################################
