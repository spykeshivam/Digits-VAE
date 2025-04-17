import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

from models.vae import VAE
from train.train_vae import train, test, prepare_data
from utils.model_analysis import visualize_latent_space, interpolate_latent_space, predict_new


def main():
    # Parameters
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 1e-2
    num_epochs = 50
    latent_dim = 16
    hidden_dim = 512
    input_dim = 784  # 28x28 MNIST images
    
    # Setup device and data loaders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, test_loader = prepare_data(batch_size)
    
    # Initialize model, optimizer, and tensorboard writer
    model = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    writer = SummaryWriter(f'runs/mnist/vae_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params:,}")
    print(model)
    
    # Training loop
    prev_updates = 0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        prev_updates = train(model, train_loader, optimizer, prev_updates, 
                            device, batch_size, writer=writer)
        test(model, test_loader, prev_updates, device, latent_dim, writer=writer)
    
    # Save the model
    torch.save(model.state_dict(), f'vae_mnist_{latent_dim}d.pt')
    
    # Visualize latent space (only for 2D latent spaces)
    if latent_dim == 2:
        visualize_latent_space(model, test_loader, device, latent_dim)
    
    # Create an interpolation in the latent space
    interpolate_latent_space(model, device, latent_dim)
    
    print("Training completed!")





if __name__ == "__main__":
    main()
    predict_new()