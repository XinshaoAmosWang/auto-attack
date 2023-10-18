from data.cifar10 import load_cifar10_test_x_y


def test_cifar10_test_x_y():
    x_tensor, y_tensor = load_cifar10_test_x_y()
    print(x_tensor.shape)
    print(y_tensor.shape)
