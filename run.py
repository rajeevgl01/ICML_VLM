import subprocess
import os
import wandb

def run_sweep():
    wandb.init()

    # Define the command to run torchrun with trainer.py
    command = [
        "torchrun",
        "--nproc_per_node=2",  # Number of GPUs or processes per node
        "trainer.py",  # Your training script
        "--train_file", "/home/local/ASURITE/rgoel15/ICML_VLM/passage_embeddings.pt",  # Specify the paths to data
        "--test_file", "/home/local/ASURITE/rgoel15/ICML_VLM/passage_embeddings_test.pt",
        "--num_epochs", f"{wandb.config.num_epochs}",  # Number of epochs
        "--lr", f"{wandb.config.lr}",  # Learning rate
        "--warmup_epochs", f"{wandb.config.warmup_epochs}" , # Warmup epochs
        "--min_lr", f"{wandb.config.min_lr}",  # Minimum learning rate
        "--weight_decay", f"{wandb.config.weight_decay}"  # Weight decay
    ]

    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = "0,1"  # Set to the GPUs you want to use (e.g., GPUs 0 and 1)

    # Run the command with the modified environment
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=my_env)

    # Print the output and errors from subprocess
    print("Output:", process.stdout)
    print("Errors:", process.stderr)


if __name__ == "__main__":
    run_sweep()
