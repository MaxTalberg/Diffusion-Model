from training import train
from config import setup_environment


def main(config_path: str, quick_test: bool = False):
    config, ddpm, optim, train_dataloader, _, accelerator, real_images = setup_environment(config_path)
    
    # Execute the training loop
    train(config, ddpm, optim, train_dataloader, accelerator, real_images, quick_test)

if __name__ == "__main__":
    main(config_path="config.yaml", 
         quick_test=False)
