import subprocess
import time
import os

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# Path to the main model script
model_script = os.path.join(project_root, "model_architecture", "main.py")

# List of model types and whether they require pre-training
model_types = {
    "semi_supervised": True,
    "carsault": False,
    "small_dilation": False,
    "multi_dilation": False
}

def run_model(model_type, pretraining_required, epochs=1000):
    """Runs the main model script with the specified model type."""
    if not os.path.exists(model_script):
        print(f"Error: {model_script} not found!")
        return

    datasets = {
        "majmin": 28,
        # "majmin7": 54, 
        # "majmininv": 73, 
        # "majmin7inv": 157 # Requires further processing, currently fails in training
    }

    for dataset, num_classes in datasets.items():
        cmd = [
            "python", model_script,
            "--model_name", model_type,
            "--model_type", model_type,
            "--data_type", dataset,
            "--num_classes", str(num_classes),
            "--epochs", str(epochs)
        ]
        
        if pretraining_required:
            cmd.extend(["--pretrain_epochs", str(epochs)])
            
        print(f"Running {model_type} model on {dataset} dataset...")
        # Run the command from the project root directory
        subprocess.run(cmd, cwd=project_root)
        time.sleep(2)  # Small pause between runs

# Run all models sequentially
for model_type, pretraining_required in model_types.items():
    # Using 10 epochs for initial testing, 1000 epochs will be used in final training
    run_model(model_type, pretraining_required, 10)
    time.sleep(2)  # Small pause between different model types