from training import train
from config import setup_environment


def main(config_path: str, fid_score: bool = False, cold_diff: bool = False, quick_test: bool = False):
    config, ddpm, optim, train_dataloader, _, accelerator, real_images = setup_environment(config_path)
    
    # Execute the training loop
    train(config, ddpm, optim, train_dataloader, accelerator, real_images, fid_score, cold_diff, quick_test)

if __name__ == "__main__":
    main(config_path="config.yaml",
         fid_score=False, 
         cold_diff=True,
         quick_test=False)
