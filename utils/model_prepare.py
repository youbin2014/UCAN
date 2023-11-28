from old_codes.resnet import *
import torchvision.transforms as transforms
import torchvision

def dataset_prepare(dataset):
    if dataset=='CIFAR10':
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=512, shuffle=True, num_workers=0)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=512, shuffle=False, num_workers=0)
    elif dataset=='ImageNet':
        transform_train_ImageNet = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test_ImageNet = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])

    return trainset,testset,trainloader,testloader

def model_prepare(dataset,model,model_path):
    if dataset=='CIFAR10':
        num_classes=10
    elif dataset=='ImageNet':
        num_classes=1000
    if model=='ResNet101':
        model = ResNet101()
    elif model=='ResNet50':
        model = ResNet50(num_classes=num_classes)
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model.cuda()
    return model
