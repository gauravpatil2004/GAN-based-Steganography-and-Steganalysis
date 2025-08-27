from data_loader import get_dataloader

# Initialize dataloader
loader = get_dataloader(batch_size=8)

# Get one batch
images, labels = next(iter(loader))

print("Batch of images shape:", images.shape)
print("Batch of labels:", labels)
