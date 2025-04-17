# MNIST Variational Autoencoder (VAE)

This project implements a Variational Autoencoder for the MNIST dataset using PyTorch. The VAE learns to generate new handwritten digits by encoding the input into a latent space and then decoding it back to image space.

## Project Structure

```
├── main.py                 # Main entry point
├── models/
│   └── vae.py              # VAE model definition
├── train/
│   └── train_vae.py        # Training functionality
└── utils/
    └── model_analysis.py   # Utilities for model analysis
```

## Requirements

- Python 3.6+
- PyTorch
- TorchVision
- NumPy
- Matplotlib
- tqdm
- TensorBoard

## Installation

```bash
# Clone the repository
git clone https://github.com/spykeshivam/mnist-vae.git
cd mnist-vae

# Install requirements
pip install torch torchvision numpy matplotlib tqdm tensorboard
```

## Usage

### Training

```bash
python main.py
```

The script will:
1. Download the MNIST dataset
2. Initialize a VAE model
3. Train the model for the specified number of epochs
4. Save the trained model as `vae_mnist_Xd.pt` (where X is the latent dimension)
5. Visualize the latent space (if latent_dim=2)
6. Generate interpolations in the latent space

### Configuration

You can adjust the following parameters in `main.py`:

- `batch_size` (default: 128)
- `learning_rate` (default: 1e-3)
- `weight_decay` (default: 1e-2)
- `num_epochs` (default: 50)
- `latent_dim` (default: 16) - Dimension of the latent space
- `hidden_dim` (default: 512) - Dimension of the hidden layers

## Model Architecture

The VAE consists of:
- An encoder network that transforms input images into parameters of a latent distribution
- A latent space where points are sampled using the reparameterization trick
- A decoder network that reconstructs images from latent vectors

## Features

- Training and evaluation loops with progress bars
- TensorBoard integration for monitoring training metrics
- Visualization tools for the latent space
- Latent space interpolation for generating transitions between digits
- New digit prediction

## TensorBoard Visualization

```bash
tensorboard --logdir=runs
```

This will show:
- Training and testing losses
- Original images vs. reconstructions
- Random samples from the latent space
- Gradient norms during training

## License

[MIT License](LICENSE)