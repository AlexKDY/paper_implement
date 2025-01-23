import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse, random
from model import Generator, Discriminator
import torch.nn as nn
import torch.optim as optim
import os

torch.backends.cudnn.benchmark = True

manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def get_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 GAN Hyperparameter Parser")
    parser.add_argument("--z_dim", type=int, default=100, help="Size of the noise vector (default: 100)")
    parser.add_argument("--img_channels", type=int, default=3, help="Number of image channels (default: 3 for RGB)")
    parser.add_argument("--feature_size", type=int, default=64, help="Feature size for Generator and Discriminator (default: 64)")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizers (default: 0.0002)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training (default: 128)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use (default: cuda)")

    return parser.parse_args()

args = get_args()

transform = transforms.Compose([
    transforms.ToTensor(),                 
    transforms.Normalize((0.5, 0.5, 0.5), 
                         (0.5, 0.5, 0.5))
])


train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, transform=transform, download=True
)

z_dim = args.z_dim
img_channels = args.img_channels
feature_size = args.feature_size
lr = args.lr
epochs = args.epochs
batch_size = args.batch_size
device = args.device

dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def train():
    generator = Generator(z_dim, img_channels, feature_size).to(device)
    discriminator = Discriminator(img_channels, feature_size).to(device)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas = (0.5, 0.99))

    for epoch in range(epochs):
        for i, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            outputs_real = discriminator(real_images).view(-1, 1)
            d_loss_real = criterion(outputs_real, real_labels)

            z = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake_images = generator(z)
            outputs_fake = discriminator(fake_images.detach()).view(-1, 1)
            d_loss_fake = criterion(outputs_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            outputs_fake = discriminator(fake_images).view(-1, 1)
            g_loss = criterion(outputs_fake, real_labels)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], "
                      f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
    

    os.makedirs("weights", exist_ok=True) 
    torch.save(generator.state_dict(), "weights/generator.pth")
    torch.save(discriminator.state_dict(), "weights/discriminator.pth")
    print("Model weights saved successfully!")

if __name__ == '__main__':
    train()