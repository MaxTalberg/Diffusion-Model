import torch
import numpy as np
import json
import time
import os
import scipy.linalg
from torchvision import transforms
from torchvision.transforms import Lambda
from torchvision.models import inception_v3, Inception_V3_Weights


def save_training_results(config, metrics):
    results = {
        "config": config,
        "metrics": metrics
    }

    path = 'metrics/'
    filename = f"results_{int(time.time())}.json"

    with open(os.path.join(path, filename), "w") as outfile:
        json.dump(results, outfile, indent=4)

inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
inception_model.eval()  # Set to evaluation mode
torch.set_num_threads(1)

def expand_gray_to_rgb(x):
    # Expand the grayscale image to have 3 channels by repeating it across the channel dimension
    return x.expand(-1, 3, -1, -1)  # Expands the second dimension (channels) to 3

def preprocess_images_for_inception(images):

    images_cpu = images.to('cpu')

    # Resize and normalize images for Inception model
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        Lambda(expand_gray_to_rgb),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_processed = transform(images_cpu)

    return image_processed

def get_features(images, model = inception_model, batch_size=32):

    images = preprocess_images_for_inception(images)

    model.eval()
    features = []
    with torch.no_grad():
        features = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_features = model(batch)
            features.append(batch_features.cpu())
        features = torch.cat(features, 0)
    return features

def frechet_distance(real_images, generated_images, inception_model=inception_model):
    real_features = get_features(real_images, inception_model)
    gen_features = get_features(generated_images, inception_model)

    m_real = real_features.mean(0)
    m_gen = gen_features.mean(0)

    real_features -= m_real
    gen_features -= m_gen

    cov_real = real_features.T @ real_features / (real_features.size(0) - 1)
    cov_gen = gen_features.T @ gen_features / (gen_features.size(0) - 1)

    # Convert tensors to NumPy arrays for scipy.linalg.sqrtm
    cov_real_np = cov_real.cpu().detach().numpy()
    cov_gen_np = cov_gen.cpu().detach().numpy()

    # Compute square root of covariance matrices using SciPy
    sqrt_cov_real = scipy.linalg.sqrtm(cov_real_np)
    sqrt_cov_gen = scipy.linalg.sqrtm(cov_gen_np)

    # Convert back to PyTorch tensors
    sqrt_cov_real = torch.from_numpy(np.real(sqrt_cov_real)).to(real_features.device)
    sqrt_cov_gen = torch.from_numpy(np.real(sqrt_cov_gen)).to(gen_features.device)

    # Compute FID score
    fid = torch.norm(m_real - m_gen)**2 + torch.trace(cov_real + cov_gen - 2 * (sqrt_cov_real @ sqrt_cov_gen))

    return fid