import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class Generator(nn.Module):
    def __init__(self, z_dim, img_channels, f_size):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, f_size * 8, kernel_size=4, stride = 1, padding=0),
            nn.BatchNorm2d(f_size*8),
            nn.ReLU(),
            nn.ConvTranspose2d(f_size*8, f_size*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(f_size*4),
            nn.ReLU(),
            nn.ConvTranspose2d(f_size*4, f_size*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(f_size * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(f_size * 2, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, img_channels, f_size):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, f_size, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(f_size, f_size*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(f_size*2),
            nn.Conv2d(f_size * 2, f_size * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(f_size * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(f_size * 4, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).view(-1)
    