import torchvision.transforms as transforms


# preporcess for pytorch 
torch_transform_mnist=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    # transforms.Resize(64),
    ])

torch_transform_cifar10=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    # transforms.Resize(224)
    ])

torch_transform_imagenet=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    # transforms.Resize(224)
    ])

# preprocess for tensorflow
