# Applcations of Machine Learning: Diffusion Models on MNIST

## Introduction

In this project the objective is to develop a standard denoising diffusion probabilistic model and taking inspiration from the "Cold Diffusion" paper by Bansal et al., (2022) implement a custom diffusion model with a unique degradation strategy.

The repository contains a series of Python scripts that explore both stochastic and deterministic diffusion models. The following section provides instructions on how to run the `main.py` script.


by **Max Talberg**

## Running the script on Docker

### Setting Up the Project

1. **Clone the repository:**
   - Clone the repository from GitLab:
     ```bash
     git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/M2_Assessment/mt942.git
     ```

2. **Build the Docker image:**

   - Build image:
     ```bash
     docker build -t ska-project .
     ```

3. **Running the script:**

   - Run the main script:
     ```bash
     docker run -v host_directory:/app/src/plots ska-project
     ```
        - Replace `host_directory` with the path to the directory where you want to save the plots, for example: `/path/to/plots` and all the images will be saved into a folder named `plots`, information acompanying will be in the terminal output.


## Running the script locally

### Setting Up the Project

1. **Clone the repository:**
   - Clone the repository from GitLab:
     ```bash
     git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/M2_Assessment/mt942.git
     ```

2. **Set up the virtual environment:**

   - Create virtual environment:
     ```bash
     conda env create -f environment.yml
     ```
    - Activate virtual environment:
      ```bash
      conda activate m2-env
      ```
3. **Running the script:**

   - Run the main script:
     ```bash
     python src/main.py --model <model_config_file> <options>
     ```
        - Replace `<model_config_file>` with the desired model:
          - `default_model.yaml` (default)
          - `model1.yaml`
          - `model2.yaml`
          - `blur_model.yaml`
        - Additional `<options>`:
          - `--fid_score`: Calculates and plots the FID score
          - `--quick_test`: Performs a quick test run (single bath)
        - Running only `python src/main.py` will run the default model with no additional options.
    

### Notes

- Running the provided script will produce a sequence of plots in the `content` directory:
    - Grid plots of the MNIST dataset at each epoch
    - Progress plots over a range of timesteps at each epoch
    - Training loss plots
    - If specified, the FID score

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Use of generative tools

This project has utilised auto-generative tools in the development of documentation that is compatible with auto-documentation tools and the development of plotting functions.