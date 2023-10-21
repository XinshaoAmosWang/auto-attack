import torch

from data.cifar10 import load_cifar10_test_x_y
from models.Wang2023Better_WRN_28_10 import load_Wang2023Better_WRN_28_10

ExpectedOutput = torch.Tensor(
    [
        [
            -0.5038,
            -1.0584,
            -0.0404,
            2.0488,
            -0.0799,
            1.1700,
            0.2806,
            -0.6269,
            -0.3371,
            -0.8527,
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
    model = load_Wang2023Better_WRN_28_10(device=device)
    # #######################################
    # #######################################
    # try forward processing
    # Note: the output is logits
    y = model(x_test[0:1].to(device=device)).detach().cpu()
    assert torch.allclose(y, ExpectedOutput, atol=1e-3)
    # ########################################
