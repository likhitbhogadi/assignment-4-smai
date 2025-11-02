# Q2: Image Colourization using CNNs

This project implements a CNN encoder-decoder network to colorize grayscale CIFAR-10 images, treating the task as a 24-class pixel-wise classification problem.

## 🗂️ File Structure

* `dataset.py`: Defines the `CIFAR10ColorizationDataset` and `get_dataloaders` function.
* `model.py`: Defines the `ColorizationNet` (encoder-decoder) architecture.
* `utils.py`: Contains helper functions for logging images to Weights & Biases.
* `train.py`: The main script to train a single model configuration.
* `sweep.py`: The script to define and run a Weights & Biases hyperparameter sweep.
* `analytical.py`: A script to calculate the answers for the "Analytical Questions" section.
* `config.yaml`: Default hyperparameters for training.
* `requirements.txt`: Python package dependencies.

## 🚀 How to Run

1.  **Install Dependencies:**
    ```bash
    pip install torch torchvision numpy wandb tqdm pyyaml
    ```

2.  **Log in to W&B:**
    ```bash
    wandb login
    ```
    
3.  **Run a Single Training:**
    Edit `config.yaml` to set your desired hyperparameters, then run:
    ```bash
    python3 train.py
    ```

4.  **Run a Hyperparameter Sweep:**
    This will run the sweep defined in `sweep.py`, using `config.yaml` for defaults.
    ```bash
    python sweep.py
    ```
    This will output a `wandb agent` command. Copy and run that command in your terminal to start the sweep agent.

5.  **Get Analytical Answers:**
    This script computes and prints the answers for the analytical questions[cite: 226].
    ```bash
    python analytical.py
    ```