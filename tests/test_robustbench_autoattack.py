import os

import rootpath
import torch

from autoattack import AutoAttack
from data.cifar10 import load_cifar10_test_x_y
from models.Wang2023Better_WRN_28_10 import load_Wang2023Better_WRN_28_10

REPO_PATH = rootpath.detect()
log_path = os.path.join(REPO_PATH, "attack_logs")


def test_autoattack():
    # config the device
    device = "cuda:0"

    # #######################################
    # load data
    data_name = "cifar10"
    x_test, y_test = load_cifar10_test_x_y()
    n_ex = x_test.shape[0]
    print(n_ex)
    batch_size = 32
    # #######################################

    # #######################################
    # load model:
    model_name = "Wang2023Better_WRN_28_10"
    model = load_Wang2023Better_WRN_28_10(device=device)
    # #######################################

    # #######################################
    # load attack
    norm = "Linf"
    # eps is the bound on the norm of the adversarial perturbations
    epsilon = 8.0 / 255.0
    # the version of AA
    version = "standard"
    adversary = AutoAttack(
        model,
        norm=norm,  # will be passed to each attack
        eps=epsilon,  # will be passed to each attack
        log_path=log_path + "/log_aa_{}.txt".format(model_name),
        version=version,  # denoting what attacks to run: standard = all the 4 attacks
    )

    # run attack and save images
    with torch.no_grad():
        # individual version, each attack is run on all test points
        adv_complete = adversary.run_standard_evaluation(
            x_test[:n_ex],
            y_test[:n_ex],
            bs=batch_size,
            return_labels=True,
            state_path=None,
        )
        torch.save(
            {"adv_complete": adv_complete},
            "{}/adv_complete_{}_{}_{}_1_{}_eps_{:.5f}_norm{:.5f}_model{}.pth".format(
                REPO_PATH,
                "aa",
                version,
                data_name,
                adv_complete.shape[0],
                epsilon,
                norm,
                model_name,
            ),
        )
