import os
import rootpath
import torch

from data.cifar10 import load_cifar10_test_x_y
from models.resnet18 import load_resnet18
from autoattack import AutoAttack
REPO_PATH = rootpath.detect()
log_path = os.path.join(
    REPO_PATH,
    "attack_logs"
)

def test_autoattack():
    # #######################################
    # load model: forward pass 
    # from [0, 1] (NCHW format expected)
    # to Logits
    device = "cuda:0"
    model = load_resnet18(device=device)
    # #######################################

    # #######################################
    # load data
    x_test, y_test = load_cifar10_test_x_y()
    # #######################################

    # #######################################
    # load attack
    norm = 'Linf'
    # eps is the bound on the norm of the adversarial perturbations
    epsilon = 8./255.
    # the version of AA
    version = 'standard'
    adversary = AutoAttack(
        model,
        norm=norm,
        eps=epsilon,
        log_path=log_path+"/log_file.txt",
        version=version
    )
    # #######################################

    # ##########################################
    # number of testing images to run:
    n_ex = 32
    # ##########################################

    # ##########################################
    # what batch size means here?
    # tried batch size = 1, 16, 32,
    # all attacks generate the same number of images as n_ex
    # # total images = 32
    #
    #
    # Using adversary.run_standard_evaluation_individual
    # # batch size = 1
    # robust accuracy by APGD-CE 	 0.00% 	 (time attack: 16.8 s)
    # robust accuracy by APGD-T 	 0.00% 	 (time attack: 19.7 s)
    # robust accuracy by FAB-T 	 0.00% 	 (time attack: 45.3 s)
    # robust accuracy by SQUARE 	 0.00% 	 (time attack: 68.4 s)
    # # batch size = 16
    # robust accuracy by APGD-CE 	 0.00% 	 (time attack: 3.4 s)
    # robust accuracy by APGD-T 	 0.00% 	 (time attack: 3.6 s)
    # robust accuracy by FAB-T 	 0.00% 	 (time attack: 7.5 s)
    # robust accuracy by SQUARE 	 0.00% 	 (time attack: 31.9 s)
    # # batch size = 32
    # robust accuracy by APGD-CE 	 0.00% 	 (time attack: 3.0 s)
    # robust accuracy by APGD-T 	 0.00% 	 (time attack: 3.0 s)
    # robust accuracy by FAB-T 	 0.00% 	 (time attack: 6.0 s)
    # robust accuracy by SQUARE 	 0.00% 	 (time attack: 19.3 s)
    #
    #
    # batch_size = 1
    # batch_size = 16
    batch_size = 32
    # ##########################################

    # run attack and save images
    with torch.no_grad():
        # individual version, each attack is run on all test points
        adv_complete = adversary.run_standard_evaluation_individual(
            x_test[:n_ex],
            y_test[:n_ex],
            bs=batch_size
        )

    # checking the norm difference
    print(adv_complete["square"].shape)
    norm_diff = torch.sum(
        torch.abs(adv_complete["square"] - x_test[:n_ex]),
        [1, 2, 3]
    )
    print(norm_diff)

    # visualise the difference
    # Revisit: x format:  [0, 1] (NCHW format expected)
    # I guess the adv format is the same
    #
    # import matplotlib.pyplot as plt
    # plt.imshow(  adv_complete["fab-t"][9].permute(1, 2, 0)  )
    # plt.show()
    # plt.imshow(  x_test[9].permute(1, 2, 0)  )
    # plt.show()
    # #