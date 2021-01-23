from __future__ import print_function
#%matplotlib inline
import argparse
import shutil
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from model import Generator, Discriminator, weights_init
from dataset import generate_dataloader, get_device, plot_imgs

# Uncomment when using FlyAI GPU for training
# from flyai.train_helper import upload_data, download 

# Upload training data to online GPU of FlyAI
'''
try:
    upload_data("./data/cats.zip", dir_name="/data", overwrite=True)
except:
    pass
'''
# download dataset for FlyAI online GPU training 
# download("/data/cats.zip", decompression=True)


# add arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=16)
parser.add_argument('--imgSize', type=int, default=64)
parser.add_argument('--nz', type=int, default=100, help='Size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64, help='Size of feature map in generator')
parser.add_argument('--ndf', type=int, default=64, help='Size of feature map in discriminator')
parser.add_argument('--epoch', type=int, default=15, help='Number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for adam. default=0.5')
parser.add_argument('--dataRoot', default='data/', help='Folder to train data. Please specify this to your own folder of training data')
parser.add_argument('--outdir', default='./output', help='Folder to output images and model checkpoints')
parser.add_argument('--ngpu', default=1, help='Number of GPUs available. Use 0 for CPU mode.')
opt = parser.parse_args()

def plot_loss():
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_real_fake_img(img_list, dataloader):
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], 
    padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()

# Generate and save fake images
def generate_fake_imgs(netG, dir_path, device, batch_size=50, img_size=50):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    for n_batch in range(0, img_size, batch_size):
        gen_z = torch.randn(batch_size, 100, 1, 1, device=device)
        gen_img = netG(gen_z)
        img = gen_img.to("cpu").clone().detach()
        img = img.numpy().transpose(0, 2, 3, 1)
        for n_img in range(gen_img.size(0)):
            vutils.save_image(
                gen_img[n_img, :, :, :], 
                os.path.join(dir_path, f'image_{n_batch + n_img:05d}.png')
            )
    shutil.make_archive('images', 'zip', dir_path)


# (hyper)parameters
ndf = opt.ndf
ngf = opt.ngf
batch_size = opt.batchSize
beta1 = opt.beta1
nz = opt.nz
epochs = opt.epoch # free to play with it
lr = opt.lr
ngpu = opt.ngpu
img_size = opt.imgSize
data_root = opt.dataRoot
output_dir = opt.outdir

pause_time = 10
real_label = 1.
fake_label = 0.

# get device and dataloader
device = get_device(ngpu=ngpu)
dataloader = generate_dataloader(batch_size, img_size, data_root)

# create Generator through Generator class and apply w_init
netG = Generator(ngf, nz).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
netG.apply(weights_init)

# create Discriminator with the smae approach
netD = Discriminator(ndf).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
netD.apply(weights_init)

# apply BCELoss function and use Adam as optimizer
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
fixed_noise = torch.randn(25, nz, 1, 1, device=device)

G_losses = []
D_losses = []
img_list = []
epoch_time = []
step = 0

print("<---------Training Images------------>")
plot_imgs(device, batch_size, img_size, data_root)

print("<---------Start Training------------>")
for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):

        #---------------TRAIN D-----------------#
        # train real imgs
        netD.zero_grad() # clear gradients
        real_img = data[0].to(device)
        b_size = real_img.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(real_img).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train fake imgs
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        #------------ TRAIN G--------------------#
        netG.zero_grad()
        label.fill_(real_label) 
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (step % 500 == 0) or ((epoch == epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        step += 1

print("<---------Loss Graph------------>")
plot_loss()
time.sleep(pause_time)

print("<---------Fake Image------------>")
plot_real_fake_img(img_list, dataloader)
time.sleep(pause_time)

print("<---------Start Generating------------>")
generate_fake_imgs(netG, output_dir, device=device)


