import torch
from tqdm import tqdm
from torchvision.transforms import v2
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def prepare_data(batch_size=128):
    """
    Prepares the MNIST dataset for training and testing.
    
    Args:
        batch_size (int): Batch size for the data loaders.
        
    Returns:
        tuple: Training and testing data loaders.
    """
    transform = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Lambda(lambda x: x.view(-1) - 0.5),
    ])

    # Download and load the training data
    train_data = torchvision.datasets.MNIST(
        '~/.pytorch/MNIST_data/', 
        download=True, 
        train=True, 
        transform=transform,
    )
    # Download and load the test data
    test_data = torchvision.datasets.MNIST(
        '~/.pytorch/MNIST_data/', 
        download=True, 
        train=False, 
        transform=transform,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True,
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=False,
    )
    
    return train_loader, test_loader


def train(model, dataloader, optimizer, prev_updates, device, batch_size, writer=None):
    """
    Trains the model on the given data.
    
    Args:
        model (nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The data loader.
        optimizer: The optimizer.
        prev_updates (int): Number of previous updates.
        device (torch.device): Device to train on.
        batch_size (int): Batch size.
        writer (SummaryWriter, optional): TensorBoard writer.
        
    Returns:
        int: Number of updates.
    """
    model.train()  # Set the model to training mode
    
    for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc="Training")):
        n_upd = prev_updates + batch_idx
        
        data = data.to(device)
        
        optimizer.zero_grad()  # Zero the gradients
        
        output = model(data)  # Forward pass
        loss = output.loss
        
        loss.backward()
        
        if n_upd % 100 == 0:
            # Calculate and log gradient norms
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
        
            print(f'Step {n_upd:,} (N samples: {n_upd*batch_size:,}), Loss: {loss.item():.4f} '
                  f'(Recon: {output.loss_recon.item():.4f}, KL: {output.loss_kl.item():.4f}) '
                  f'Grad: {total_norm:.4f}')

            if writer is not None:
                global_step = n_upd
                writer.add_scalar('Loss/Train', loss.item(), global_step)
                writer.add_scalar('Loss/Train/BCE', output.loss_recon.item(), global_step)
                writer.add_scalar('Loss/Train/KLD', output.loss_kl.item(), global_step)
                writer.add_scalar('GradNorm/Train', total_norm, global_step)
            
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)    
        
        optimizer.step()  # Update the model parameters
        
    return prev_updates + len(dataloader)


def test(model, dataloader, cur_step, device, latent_dim, writer=None):
    """
    Tests the model on the given data.
    
    Args:
        model (nn.Module): The model to test.
        dataloader (torch.utils.data.DataLoader): The data loader.
        cur_step (int): The current step.
        device (torch.device): Device to test on.
        latent_dim (int): Dimensionality of the latent space.
        writer (SummaryWriter, optional): TensorBoard writer.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0
    
    with torch.no_grad():
        for data, _ in tqdm(dataloader, desc='Testing'):
            data = data.to(device)
            
            output = model(data, compute_loss=True)  # Forward pass
            
            test_loss += output.loss.item()
            test_recon_loss += output.loss_recon.item()
            test_kl_loss += output.loss_kl.item()
            
    test_loss /= len(dataloader)
    test_recon_loss /= len(dataloader)
    test_kl_loss /= len(dataloader)
    print(f'====> Test set loss: {test_loss:.4f} (BCE: {test_recon_loss:.4f}, KLD: {test_kl_loss:.4f})')
    
    if writer is not None:
        writer.add_scalar('Loss/Test', test_loss, global_step=cur_step)
        writer.add_scalar('Loss/Test/BCE', test_recon_loss, global_step=cur_step)
        writer.add_scalar('Loss/Test/KLD', test_kl_loss, global_step=cur_step)
        
        # Log reconstructions
        recon_images = (output.x_recon.view(-1, 1, 28, 28) + 0.5).clamp(0, 1)
        orig_images = (data.view(-1, 1, 28, 28) + 0.5).clamp(0, 1)
        writer.add_images('Test/Reconstructions', recon_images[:16], global_step=cur_step)
        writer.add_images('Test/Originals', orig_images[:16], global_step=cur_step)
        
        # Log random samples from the latent space
        z = torch.randn(16, latent_dim).to(device)
        samples = model.decode(z)
        sample_images = (samples.view(-1, 1, 28, 28) + 0.5).clamp(0, 1)
        writer.add_images('Test/Samples', sample_images, global_step=cur_step)

