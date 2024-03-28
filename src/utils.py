import os
import json
import time
import torch
import numpy as np
import scipy.linalg
from torchvision import transforms
from torchvision.transforms import Lambda
from torchvision.models import inception_v3, Inception_V3_Weights

# Initialise inception model and set to evaluation mode for FID calculation
inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
inception_model.eval()


def save_training_results(config, config_model, metrics):
    """
    Save training results including configuration and metrics to a JSON file.

    Parameters
    ----------
    config : dict
        Configuration parameters used for training, such as learning rate,
        batch size, etc.
    metrics : dict
        Training metrics such as loss and accuracy collected during training.

    Returns
    -------
    None
    """
    results = {"config": config, "model": config_model, "metrics": metrics}

    path = "metrics/"
    filename = f"results_{int(time.time())}.json"

    with open(os.path.join(path, filename), "w") as outfile:
        json.dump(results, outfile, indent=4)


def expand_gray_to_rgb(x):
    """
    Expand a grayscale image to have 3 channels by repeating it across the
    channel dimension.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor representing a batch of grayscale images.

    Returns
    -------
    torch.Tensor
        The expanded tensor with 3 channels for each image.
    """
    # Expands the to 3 channels by repeating the grayscale channel
    return x.expand(-1, 3, -1, -1)


def preprocess_images_for_inception(images):
    """
    Preprocess a batch of images for the Inception model.

    Parameters
    ----------
    images : torch.Tensor
        A tensor representing a batch of images.

    Returns
    -------
    torch.Tensor
        The processed image tensor ready for input to the Inception model.
    """
    images_cpu = images.to("cpu")

    # Resizing, expanding gray to RGB, and normalisation
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            Lambda(expand_gray_to_rgb),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Apply transformations to the images
    image_processed = transform(images_cpu)

    return image_processed


def get_features(images, model=inception_model, batch_size=16):
    """
    Extract features from images using a pretrained model, batch by batch.

    Parameters
    ----------
    images : torch.Tensor
        A tensor representing a batch of images to process.
    model : torch.nn.Module, optional
        The model to use for feature extraction, default is the global
        `inception_model`.
    batch_size : int, optional
        The size of each batch to process at a time

    Returns
    -------
    torch.Tensor
        A tensor of extracted features from the images.
    """
    images = preprocess_images_for_inception(images)

    model.eval()
    features = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            batch_features = model(batch)
            features.append(batch_features.cpu())
        features = torch.cat(features, 0)
    return features


def frechet_distance(real_images, generated_images, inception_model=inception_model):
    """
    Calculate the Frechet Inception Distance (FID) between real and
    generated images.

    Parameters
    ----------
    real_images : torch.Tensor
        A tensor of real images.
    generated_images : torch.Tensor
        A tensor of generated images to compare against the real images.
    inception_model : torch.nn.Module, optional
        The Inception model to use for feature extraction,
        default is the global `inception_model`.

    Returns
    -------
    float
        The calculated Frechet Inception Distance.
    """
    real_features = get_features(real_images, inception_model)
    gen_features = get_features(generated_images, inception_model)

    # Compute mean feature vectors
    m_real = real_features.mean(0)
    m_gen = gen_features.mean(0)

    # Center the feature vectors
    real_features -= m_real
    gen_features -= m_gen

    # Compute the covariance matrices
    cov_real = real_features.T @ real_features / (real_features.size(0) - 1)
    cov_gen = gen_features.T @ gen_features / (gen_features.size(0) - 1)

    cov_real_np = cov_real.cpu().detach().numpy()
    cov_gen_np = cov_gen.cpu().detach().numpy()

    # Compute the square root of the covariance matrices
    sqrt_cov_real = scipy.linalg.sqrtm(cov_real_np)
    sqrt_cov_gen = scipy.linalg.sqrtm(cov_gen_np)

    # Convert back to tensors
    sqrt_cov_real = torch.from_numpy(np.real(sqrt_cov_real)).to(real_features.device)
    sqrt_cov_gen = torch.from_numpy(np.real(sqrt_cov_gen)).to(gen_features.device)

    # Final FID score computation
    fid = torch.norm(m_real - m_gen) ** 2 + torch.trace(
        cov_real + cov_gen - 2 * (sqrt_cov_real @ sqrt_cov_gen)
    )

    return fid
