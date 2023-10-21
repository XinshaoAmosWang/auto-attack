import torch

from data.cifar10 import load_cifar10_test_x_y
from robustbench.utils import load_model

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
    model = load_model(
        model_name="Wang2023Better_WRN-70-16",
        # TODO: could be confirmed with other examples without reading the code
        # is the threat model necessary, with other objectives
        # or the only purpose is to create a sub folder type?
        # 1.  model_dir_ = Path(model_dir) / dataset_.value / threat_model_.value
        # 2. models = all_models[dataset_][threat_model_]
        threat_model="Linf",
    )
    model.to(device=device)
    model.eval()
    # #########################################
    # the final layer of model:
    # (logits): Linear(in_features=1024, out_features=10, bias=True)
    # the first layer of model:
    # (init_conv):
    # Conv2d(
    #   3, 16,
    #   kernel_size=(3, 3),
    #   stride=(1, 1),
    #   padding=(1, 1), bias=False
    # )
    #
    # #######################################

    # #######################################
    # try forward processing
    # Note: the output is logits
    # detach(): remove grad_fn=<ToCopyBackward0>
    # cpu(): from gpu to cpu
    y = model(x_test[0:1].to(device=device)).detach().cpu()
    assert torch.allclose(y, ExpectedOutput, atol=1e-3)
    # ########################################
