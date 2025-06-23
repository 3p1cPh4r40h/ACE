import os
import subprocess
import time

# Configuration
EPOCHS = 10  # Number of epochs for training

def run_models():
    """Run all models with the specified number of epochs."""
    print(f"\n{'='*80}")
    print(f"Running all models with {EPOCHS} epochs")
    print(f"{'='*80}")
    
    # Run run_models.py with batch mode
    subprocess.run(['python', 'run_models.py', '--batch_mode', '--epochs', str(EPOCHS)])
    
    print("\nTraining completed. Waiting 5 seconds before testing...")
    time.sleep(5)

def test_models():
    """Test all trained models."""
    print(f"\n{'='*80}")
    print("Testing all models")
    print(f"{'='*80}")
    
    # Run test_models.py
    subprocess.run(['python', 'test_models.py'])

def main():
    """Main function to run and test all models."""
    try:
        # Run all models
        run_models()
        
        # Test all models
        test_models()
        
        print("\nAll operations completed successfully!")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

if __name__ == "__main__":
    main() 