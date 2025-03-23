import subprocess
import time
from datetime import datetime
import os

# Path to the models folder
models_folder = "models"

# List of your model script filenames
model_scripts = [
    "CarsaultACEModel.py",
    "SmallDilationACEModel.py"
]

def run_model(script_name):
    log_filename = f"logs\{script_name}_log.txt"

    """Runs a Python script from the models folder and logs its output with timestamps."""
    script_path = os.path.join(models_folder, script_name)
    
    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found!")
        return

    start_time = datetime.now()
    print(f"Starting {script_name} at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    with open(log_filename, "a") as log_file:
        log_file.write(f"\n===== Running {script_name} at {start_time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")

        # Run the script and capture output
        process = subprocess.Popen(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Stream output to log with timestamps
        for line in process.stdout:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"[{timestamp}] {line}"
            print(log_entry, end="")  # Print to console
            log_file.write(log_entry)  # Save to file

        # Capture any errors
        stderr = process.stderr.read()
        if stderr:
            error_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_file.write(f"\n[{error_timestamp}] ERROR:\n{stderr}\n")
            print(f"\n[{error_timestamp}] ERROR:\n{stderr}\n")

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Finished {script_name} at {end_time.strftime('%Y-%m-%d %H:%M:%S')} (Duration: {duration})\n")
    with open(log_filename, "a") as log_file:
        log_file.write(f"\n===== Finished {script_name} at {end_time.strftime('%Y-%m-%d %H:%M:%S')} (Duration: {duration}) =====\n\n")

# Run all models sequentially
for script in model_scripts:
    run_model(script)
    time.sleep(2)  # Small pause between scripts (optional)