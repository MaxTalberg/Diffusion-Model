from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_dataloaders(batch_size, num_workers):
    """
    Prepare dataloaders for the MNIST dataset for both training and testing.

    This function initializes the MNIST dataset with a standard preprocessing pipeline,
    which includes converting images to tensors and normalizing them. It then creates
    dataloaders for both the training and testing sets of the MNIST dataset, which are
    used to iterate over the dataset in batches during model training and evaluation.

    Parameters
    ----------
    batch_size : int
        The number of samples in each batch of data.
    num_workers : int
        The number of subprocesses to use for data loading. 0 means that the data will
        be loaded in the main process.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        A tuple containing the training and testing dataloaders for the MNIST dataset.
        - The first element is the DataLoader for the training set.

    """
    # Define the image transformations: Convert images to PyTorch tensors and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.0))  # Normalize images with mean=0.5, std=1.0
    ])

    # Initialize the MNIST training dataset with specified transformations
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)

    # Create the DataLoader for the training dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    return dataloader
