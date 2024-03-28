# Applcations of Machine Learning: Diffusion Models on MNIST

## Introduction

In this project the objective is to develop a standard denoising diffusion probabilistic model and taking inspiration from the "Cold Diffusion" paper by Bansal et al., (2022) implement a custom diffusion model with a unique degradation strategy.

The repository contains a series of Python scripts that explore both stochastic and deterministic diffusion models. The following section provides instructions on how to run the `main.py` script.


by **Max Talberg**

## Running the script locally

### Setting Up the Project

1. **Clone the repository:**
   - Clone the repository from GitLab:
     ```bash
     git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/M2_Assessment/mt942.git
     ```

2. **Set up the virtual environment:**
   - Navigate to the project directory:
     ```bash
       cd mt942
       ```
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
     python src/main.py --model <model_config> <options>
     ```
        - Replace `<model_config>` with the desired model:
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


## Running the script on Docker

### Setting Up the Project

1. **Clone the repository:**
   - Clone the repository from GitLab:
     ```bash
     git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/M2_Assessment/mt942.git
     ```

2. **Build the Docker image:**
   - Navigate to the project directory:
     ```bash
       cd mt942
       ```
   - Build image:
     ```bash
     docker build -t diffusion-model .
     ```

3. **Running the script:**

   - Run the main script:
     ```bash
     docker run -v $(pwd)/output:/app/output diffusion-model src/main.py --model <model_config> <options>
     ```
        - Replace `<model_config>` with the desired model:
          - `default_model.yaml` (default)
          - `model1.yaml`
          - `model2.yaml`
          - `blur_model.yaml`
        - Additional `<options>`:
          - `--fid_score`: Calculates and plots the FID score
          - `--quick_test`: Performs a quick test run (single bath)
        - Running only `docker run my-diffusion-model src/main.py` will run the default model with no additional options.
        - The images are mounted onto the current working directory into an 'output` folder.
        - The diffusion model runs a lot slower in Docker than the local script. As such I would recommend running the script locally or using `--quick_test` to speed up the process.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Use of generative tools

This project has utilised auto-generative tools in the development of documentation that is compatible with auto-documentation tools, latex formatting and the development of plotting functions. 

Example prompts used for this project:
- Generate doc-strings in NumPy format for this function.
- Generate Latex code for an algorithm.
- Generate Latex code for an equation.
- Generate Python code for a 9 by 1 subplot.
