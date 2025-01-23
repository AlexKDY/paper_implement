import matplotlib.pyplot as plt
import numpy as np
import argparse
from model import Generator
import torch

def get_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 GAN Hyperparameter Parser")
    parser.add_argument("--z_dim", type=int, default=100, help="Size of the noise vector (default: 100)")
    parser.add_argument("--img_channels", type=int, default=3, help="Number of image channels (default: 3 for RGB)")
    parser.add_argument("--feature_size", type=int, default=64, help="Feature size for Generator and Discriminator (default: 64)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use (default: cuda)")
    return parser.parse_args()

args = get_args()
z_dim = args.z_dim
img_channels = args.img_channels
feature_size = args.feature_size
device = args.device

def generate_images():
    generator = Generator(z_dim, img_channels, feature_size).to(device)
    generator.load_state_dict(torch.load("weights/generator.pth"))
    generator.eval()  

    num_images = 16      
    z = torch.randn(num_images, z_dim, 1, 1).to(device)
    
    with torch.no_grad():  
        fake_images = generator(z).cpu().numpy()
    
    fake_images = (fake_images + 1) / 2  
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        img = np.transpose(fake_images[i], (1, 2, 0)) 
        ax.imshow(img)
        ax.axis("off")
    plt.show()

if __name__ == '__main__':
    generate_images()