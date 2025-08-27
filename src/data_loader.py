import torch
from torchvision import datasets, transforms

def get_dataloader(batch_size=64):
    # Step 1: Define image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),                       # Convert image to tensor
        transforms.Normalize((0.5,), (0.5,))         # Normalize to range [-1, 1]
    ])

    # Step 2: Download CIFAR-10 dataset
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

    # Step 3: Create DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
