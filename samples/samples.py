import torchvision

data_path = '/tf/datasets'

mnist = torchvision.datasets.MNIST(root=data_path, train=False,
                                       download=False)

cifar10 = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                       download=False)

imagenet = torchvision.datasets.ImageNet(root=data_path, split='val')
