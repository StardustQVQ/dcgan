import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# Generater a dataset loader. Subdirectories are required in root directory.
def generate_dataloader(batch_size, img_size, data_root):

    n_workers = 2

    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    train_data = dset.ImageFolder(data_root, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        train_data, 
        shuffle=True, 
        batch_size=batch_size, 
        num_workers=n_workers,
        drop_last=True
    )

    return dataloader

# Return a training device chooser
def get_device(ngpu):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    return device

# Plot training images
def plot_imgs(device, batch_size, img_size, data_root):
    dataloader = generate_dataloader(batch_size, img_size, data_root)
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))


