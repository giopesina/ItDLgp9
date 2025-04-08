import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
latent_dim = 100
batch_size = 16
lr = 0.001
num_epochs = 10
image_size = 128

from torchvision.datasets import ImageFolder
from torchvision.io import read_image

# Change this to your dataset folder path
data_path = "/workspaces/ItDLgp9/output/00021_CYCLEGAN_BRAIN_MRI_T1_T2"

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.Grayscale(num_output_channels=1),  # Only if your model expects 1 channel
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
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.disc(x)           # shape: [batch_size, 1, H, W]
        return x.mean([2, 3]).view(-1)  # shape: [batch_size]

# Initialize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)
netD = Discriminator().to(device)

criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

# Training Loop
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)

        # Train Discriminator
        netD.zero_grad()
        label_real = torch.full((batch_size,), 1., device=device)
        label_fake = torch.full((batch_size,), 0., device=device)

        output_real = netD(real_images)
        lossD_real = criterion(output_real, label_real)

        fake_images = netG(noise)
        output_fake = netD(fake_images.detach())
        lossD_fake = criterion(output_fake, label_fake)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # Train Generator
        netG.zero_grad()
        label_gen = torch.full((batch_size,), 1., device=device)  # trick discriminator
        output_gen = netD(fake_images)
        lossG = criterion(output_gen, label_gen)
        lossG.backward()
        optimizerG.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(dataloader)} \
                  Loss D: {lossD:.4f}, loss G: {lossG:.4f}")

    # Generate sample images


import os
from torchvision.utils import save_image

# Create a folder for saving images (only once)
os.makedirs("generated_images", exist_ok=True)

# Generate and save sample images
with torch.no_grad():
    fake = netG(fixed_noise).detach().cpu()

# Normalize from [-1, 1] back to [0, 1] and save
save_image(fake, f"generated_images/epoch_{epoch+1:03d}.png", normalize=True)

# Optionally also show the image inline
grid = torchvision.utils.make_grid(fake, normalize=True)
plt.figure(figsize=(8, 8))
plt.imshow(np.transpose(grid, (1, 2, 0)))
plt.title(f"Generated Images - Epoch {epoch + 1}")
plt.axis("off")
plt.show()

