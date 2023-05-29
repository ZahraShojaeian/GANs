# -*- coding: utf-8 -*-


import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST # Training dataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
#import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#torch.manual_seed(0) 
#torch.cuda.empty_cache()

### Set the device to use CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Define hyperparameters

z_dim = 100
hidden_dim_g = 256
hidden_dim_d = 32
image_chan = 1
num_epochs = 50
batch_size = 64
learning_rate = 2e-4

### Define data transformation for MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


### Load MNIST dataset as tensors
dataloader = DataLoader(
    MNIST(root="dataset/",download=True, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True
)

### Generator model
class Generator(nn.Module):
    def __init__(self, z_dim,hidden_dim_g):
        super(Generator, self).__init__()
        ### Define the generator layers
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, hidden_dim_g*4,3,2),
            nn.BatchNorm2d(hidden_dim_g*4),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim_g*4,hidden_dim_g*2,4,1),
            nn.BatchNorm2d(hidden_dim_g*2),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim_g*2,hidden_dim_g,3,2),
            nn.BatchNorm2d(hidden_dim_g),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim_g ,1,4,2),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)

### Discriminator model
class Discriminator(nn.Module):
    def __init__(self, image_chan, hidden_dim_d):
        super(Discriminator, self).__init__()
        ### Define the discriminator layers
        self.disc = nn.Sequential(
            nn.Conv2d(image_chan,hidden_dim_d,4,2),
            nn.BatchNorm2d(hidden_dim_d),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim_d,hidden_dim_d*2, 4, 1),
            nn.BatchNorm2d(hidden_dim_d*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim_d*2,hidden_dim_d*4, 4, 2),
            nn.BatchNorm2d(hidden_dim_d*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim_d*4, 1,4, 1),
            #nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


### Create generator and discriminator models
gen = Generator(z_dim, hidden_dim_g).to(device)
disc = Discriminator(image_chan, hidden_dim_d).to(device)

### Define loss criterion
criterion =  nn.BCEWithLogitsLoss()

### Define optimizers for generator and discriminator
optimizer_g = torch.optim.Adam(gen.parameters(), lr=learning_rate,betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(disc.parameters(), lr=learning_rate,betas=(0.5, 0.999))

### Initialize the weights of the models
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
gen = gen.apply(weights_init)
disc = disc.apply(weights_init)

### Discriminator loss calculation
def Discriminator_loss(gen, disc, criterion, real, num_images, z_dim, device):
  ### Generate random noise
    noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)
    #print(noise.shape)
  ### Generate fake images using the generator
    fake = gen(noise)
  ### Pass real images through the discriminator
    disc_real = disc(real)
  ### Calculate the loss for real images
    lossD_real = criterion(disc_real, torch.ones_like(disc_real))
  ### Pass fake images through the discriminator
    disc_fake = disc(fake.detach())
  ### Calculate the loss for fake images
    lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
  ### Calculate the average discriminator loss
    disc_loss = (lossD_real + lossD_fake)/2
    return disc_loss

### Generator loss calculation
def Generator_loss(gen, disc, criterion, num_images, z_dim, device):
    noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
    fake = gen(noise)
    disc_fake = disc(fake)
    gen_loss= criterion(disc_fake, torch.ones_like(disc_fake))    
    return gen_loss

# Step for displaying generated images and loss
display_step = 500
cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):

    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
    


### Training loop
for epoch in range(num_epochs):
    for real, _ in tqdm(dataloader):
        batch_size = len(real)
        real = real.to(device)

         ### Training the discriminator 
        # Zero out the gradients before backpropagation
        optimizer_d.zero_grad()

        # Calculate discriminator loss
        disc_loss = Discriminator_loss(gen, disc, criterion, real, batch_size, z_dim, device)

        # Update gradients
        disc_loss.backward(retain_graph=True)

        # Update optimizer
        optimizer_d.step()
        
        
        ### Training the generator
        optimizer_g.zero_grad()
        gen_loss = Generator_loss(gen, disc, criterion, batch_size, z_dim, device)
        gen_loss.backward(retain_graph=True)
        optimizer_g.step()

        ###  The average losses
        mean_discriminator_loss += disc_loss.item() / display_step
        mean_generator_loss += gen_loss.item() / display_step

        ### Visualization code ###
        if cur_step % display_step == 0 and cur_step > 0:
            print(f"Epoch {epoch}, Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            fake_noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake = gen(fake_noise)
            show_tensor_images(fake)
            show_tensor_images(real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1
