import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from models.vae import VAE
from tqdm import tqdm

def visualize_latent_space(model, dataloader, device, latent_dim):
    """
    Visualizes the latent space of the VAE.
    
    Args:
        model (nn.Module): The trained VAE model.
        dataloader (torch.utils.data.DataLoader): The data loader.
        device (torch.device): Device to run on.
        latent_dim (int): Dimensionality of the latent space.
    """
    if latent_dim != 2:
        print("Visualization only supported for 2D latent space.")
        return
    
    model.eval()
    z_samples = []
    labels = []
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Collecting latent samples"):
            data = data.to(device)
            output = model(data, compute_loss=False)
            z_samples.append(output.z_sample.cpu().numpy())
            labels.append(target.numpy())
    
    z_samples = np.concatenate(z_samples, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # 2D scatter plot colored by digit class
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_samples[:, 0], z_samples[:, 1], c=labels, cmap='tab10', 
                         alpha=0.5, s=5)
    plt.colorbar(scatter, label='Digit Class')
    plt.title('MNIST 2D Latent Space')
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.savefig('vae_mnist_latent_scatter.png')
    plt.close()
    
    # 2D histogram of the latent space
    plt.figure(figsize=(10, 8))
    plt.hist2d(z_samples[:, 0], z_samples[:, 1], bins=50, cmap='viridis', norm=plt.cm.colors.LogNorm())
    plt.colorbar(label='Count (log scale)')
    plt.title('MNIST 2D Latent Space Density')
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.savefig('vae_mnist_latent_hist.png')
    plt.close()
    
    # 1D marginals
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].hist(z_samples[:, 0], bins=50, alpha=0.7)
    axs[0].set_title('Marginal Distribution of z[0]')
    axs[0].set_xlabel('z[0]')
    axs[0].set_ylabel('Count')
    
    axs[1].hist(z_samples[:, 1], bins=50, alpha=0.7)
    axs[1].set_title('Marginal Distribution of z[1]')
    axs[1].set_xlabel('z[1]')
    axs[1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('vae_mnist_latent_marginals.png')
    plt.close()


def interpolate_latent_space(model, device, latent_dim):
    """
    Creates an interpolation in the latent space.
    
    Args:
        model (nn.Module): The trained VAE model.
        device (torch.device): Device to run on.
        latent_dim (int): Dimensionality of the latent space.
    """
    model.eval()
    
    if latent_dim == 2:
        # Create a grid interpolation
        n = 15
        z1 = torch.linspace(-0, 1, n)
        z2 = torch.zeros_like(z1) + 2
        z = torch.stack([z1, z2], dim=-1).to(device)
    else:
        # Random interpolation for high-dimensional spaces
        n = 15
        z_start = torch.randn(1, latent_dim).to(device)
        z_end = torch.randn(1, latent_dim).to(device)
        alphas = torch.linspace(0, 1, n).unsqueeze(-1).to(device)
        z = z_start * (1 - alphas) + z_end * alphas
    
    with torch.no_grad():
        samples = model.decode(z)
        samples = (samples.view(-1, 28, 28) + 0.5).clamp(0, 1)

    # Plot the generated images
    fig, axs = plt.subplots(1, n, figsize=(n, 1))
    for i in range(n):
        axs[i].imshow(samples[i].cpu().numpy(), cmap='gray')
        axs[i].axis('off')
        
    plt.savefig('vae_mnist_interp.png')
    plt.close()


def predict_new(model_path='vae_mnist_16d.pt', num_samples=20, latent_dim=16, hidden_dim=512, input_dim=784):
    """
    Load a trained VAE model and generate new images by sampling from the latent space.
    
    Args:
        model_path (str): Path to the saved VAE model.
        num_samples (int): Number of images to generate.
        latent_dim (int): Latent space dimension used in training.
        hidden_dim (int): Hidden layer size used in training.
        input_dim (int): Input dimension (e.g., 784 for 28x28 MNIST).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Recreate model and load weights
    model = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Sample from standard normal distribution
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = model.decode(z)
        samples = (samples.view(-1, 1, 28, 28) + 0.5).clamp(0, 1)  # Reshape & normalize

    # Display generated samples
    grid = torchvision.utils.make_grid(samples, nrow=num_samples, padding=2)
    plt.figure(figsize=(num_samples, 2))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title('Generated Samples from VAE')
    plt.show()
