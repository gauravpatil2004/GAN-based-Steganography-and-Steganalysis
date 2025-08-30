import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


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


def create_data_loader(batch_size=16, data_path='./data'):
    """
    Create data loader for steganography training.
    
    Args:
        batch_size: Batch size for training
        data_path: Path to store dataset
    
    Returns:
        DataLoader for steganography training
    """
    
    # Custom collate function for steganography pairs
    def stego_collate_fn(batch):
        """Create cover-secret pairs from batch"""
        covers = []
        secrets = []
        
        # Pair images: use alternating images as cover/secret
        for i in range(0, len(batch), 2):
            if i + 1 < len(batch):
                cover_img, _ = batch[i]
                secret_img, _ = batch[i + 1]
                covers.append(cover_img)
                secrets.append(secret_img)
        
        # Handle odd batch sizes
        if len(covers) == 0:
            cover_img, _ = batch[0]
            secret_img = cover_img.clone()  # Use same image as secret
            covers.append(cover_img)
            secrets.append(secret_img)
        
        return torch.stack(covers), torch.stack(secrets)
    
    # Define transforms for steganography
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Note: We'll normalize to [-1, 1] range in training loop
    ])
    
    # Load CIFAR-10 dataset
    dataset = datasets.CIFAR10(
        root=data_path, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Create DataLoader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size * 2,  # *2 because we pair images
        shuffle=True,
        collate_fn=stego_collate_fn,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=False
    )
    
    return dataloader
