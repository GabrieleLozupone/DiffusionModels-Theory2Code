import torch
from torchvision import datasets, transforms

def get_mnist_dataset(root='./data'):
    """
    Get MNIST dataset with appropriate transformations.
    
    Args:
        root (str): Root directory for dataset storage.
        
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    # Define the transformation for the dataset - resize to 32x32
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize MNIST from 28x28 to 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    # Download and load the MNIST dataset
    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset

def plot_mnist_samples(dataset, num_samples=6):
    """
    Visualize samples from MNIST dataset.
    
    Args:
        dataset: MNIST dataset
        num_samples (int): Number of samples to visualize
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 4))
    for i in range(num_samples):
        image, label = dataset[i]
        axes[i].imshow(image.squeeze(), cmap='gray')
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
