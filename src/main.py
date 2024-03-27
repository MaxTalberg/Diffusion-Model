import os
import argparse
from training import train
from config import setup_environment

# Change the working directory to the location of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def main(
    config_path: str,
    model_path: str,
    fid_score: bool = False,
    cold_diff: bool = False,
    quick_test: bool = False,
):
    (
        config,
        config_model,
        ddpm,
        optim,
        dataloader,
        accelerator,
        real_images,
    ) = setup_environment(config_path, model_path)

    # Execute the training loop
    train(
        config,
        config_model,
        ddpm,
        optim,
        dataloader,
        accelerator,
        real_images,
        fid_score,
        cold_diff,
        quick_test,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the diffusion model")
    parser.add_argument(
        "--model", default="default_model.yaml", help="Path to model file."
    )
    parser.add_argument("--fid_score", action="store_true", help="Calculate FID score.")
    parser.add_argument("--cold_diff", action="store_true", help="Use cold diffusion.")
    parser.add_argument("--quick_test", action="store_true", help="Run a quick test.")

    args = parser.parse_args()

    main(
        config_path="config.yaml",
        model_path=args.model,
        fid_score=args.fid_score,
        cold_diff=args.cold_diff,
        quick_test=args.quick_test,
    )
