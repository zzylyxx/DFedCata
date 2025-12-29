"""
Example usage of DFedCata Federated Learning Framework

This script demonstrates how to run a simple federated learning experiment
using the DFedCata algorithm on CIFAR-10 dataset with LeNet model.
"""

import subprocess
import sys
import os

def run_example():
    """Run a basic federated learning experiment."""

    # Example command for CIFAR-10 with LeNet
    cmd = [
        sys.executable, "train_1.py",
        "--dataset", "CIFAR10",
        "--model", "LeNet",
        "--total-client", "50",      # Smaller number for quick demo
        "--comm-rounds", "10",       # Fewer rounds for quick demo
        "--local-epochs", "2",       # Fewer local epochs
        "--batchsize", "64",         # Smaller batch size
        "--seed", "42"
    ]

    print("Running DFedCata example...")
    print("Command:", " ".join(cmd))
    print("="*50)

    try:
        result = subprocess.run(cmd, cwd=os.getcwd())
        if result.returncode == 0:
            print("="*50)
            print("Example completed successfully!")
        else:
            print("="*50)
            print(f"Example failed with return code: {result.returncode}")
    except KeyboardInterrupt:
        print("\nExample interrupted by user.")
    except Exception as e:
        print(f"Error running example: {e}")

if __name__ == "__main__":
    run_example()
