from training import train
from config import setup_environment


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
    main(
        config_path="config.yaml",
        model_path="default_model.yaml",
        fid_score=True,
        cold_diff=False,
        quick_test=False,
    )
