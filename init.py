# DCGAN Training Script with Mode Collapse Fixes and GIF Output
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
import imageio

# Settings
latent_dim = 100
batch_size = 16
lr = 0.001
num_epochs = int(input())
image_size = 128
data_path = "/workspaces/ItDLgp9/output/"
os.makedirs("generated_images", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Transform
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = ImageFolder(root=data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.disc(x)
        return x.view(-1)

# Init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)
netD = Discriminator().to(device)
criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
fixed_noise = torch.randn(16, latent_dim, 1, 1, device=device)

g_losses = []
d_losses = []

# Training
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)

        # Apply instance noise (non-inplace)
        real_images = real_images + 0.05 * torch.randn_like(real_images)

        # Train Discriminator
        netD.zero_grad()
        label_real = torch.full((batch_size,), 0.9, device=device)
        label_fake = torch.full((batch_size,), 0.0, device=device)

        output_real = netD(real_images)
        lossD_real = criterion(output_real, label_real)

        fake_images = netG(noise)
        fake_images = fake_images + 0.05 * torch.randn_like(fake_images)
        output_fake = netD(fake_images.detach())
        lossD_fake = criterion(output_fake, label_fake)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # Train Generator
        netG.zero_grad()
        label_gen = torch.full((batch_size,), 0.9, device=device)
        output_gen = netD(fake_images)
        lossG = criterion(output_gen, label_gen)
        lossG.backward()
        optimizerG.step()

        if i == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(dataloader)} Loss D: {lossD:.4f}, loss G: {lossG:.4f}")

    d_losses.append(lossD.item())
    g_losses.append(lossG.item())

    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    save_image(fake, f"generated_images/epoch_{epoch+1:03d}.png", normalize=True)

    if (epoch + 1) % 50 == 0:
        torch.save(netG.state_dict(), f"checkpoints/netG_epoch{epoch+1}.pth")
        torch.save(netD.state_dict(), f"checkpoints/netD_epoch{epoch+1}.pth")

# Plot loss
plt.plot(d_losses, label="Discriminator Loss")
plt.plot(g_losses, label="Generator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("DCGAN Training Loss")
plt.savefig("loss_curve01.png")
plt.close()

# Generate GIF
try:
    image_files = sorted([
        f for f in os.listdir("generated_images") if f.endswith(".png")
    ], key=lambda x: int(x.split('_')[1].split('.')[0]))

    images = [imageio.imread(os.path.join("generated_images", f)) for f in image_files]
    imageio.mimsave("dcgan_progress01.gif", images, duration=0.25)
    print("Saved GIF as dcgan_progress.gif")
except Exception as e:
    print("Failed to generate GIF:", e)

print("Training complete. Loss curve and GIF saved.")
