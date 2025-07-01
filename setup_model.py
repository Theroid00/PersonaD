#!/usr/bin/env python3
"""
Helper script to set up the model after cloning the repository.
This script will either train a new model or allow the user to place their own fine-tuned model.
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

def check_model_files():
    """Check if model files exist"""
    ft_model_path = Path("fine_tuned_model/model.safetensors")
    checkpoint_path = Path("checkpoints/best_model/model.safetensors")
    
    if ft_model_path.exists():
        print(f"Model file exists at {ft_model_path}")
        return True
    
    if checkpoint_path.exists():
        print(f"Checkpoint model file exists at {checkpoint_path}")
        return True
    
    return False

def create_model_dirs():
    """Create model directories if they don't exist"""
    os.makedirs("fine_tuned_model", exist_ok=True)
    os.makedirs("checkpoints/best_model", exist_ok=True)
    print("Created model directories")

def train_new_model(message_files, epochs, augment):
    """Train a new model using the provided message files"""
    import subprocess
    
    # Check if message files exist
    files_exist = all(os.path.exists(f) for f in message_files)
    if not files_exist:
        print("Error: One or more message files do not exist")
        print("Please place your Instagram JSON export files in the project directory")
        return False
    
    # Build command
    cmd = ["python", "clean_and_finetune.py", "--data_files"] + message_files
    cmd += ["--epochs", str(epochs)]
    if augment:
        cmd.append("--augment")
    
    # Run training
    print(f"Starting training with command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return check_model_files()
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Set up the model after cloning the repository")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--message_files", nargs="+", default=["message_1.json"], 
                        help="Message files to use for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--augment", action="store_true", help="Augment training data")
    
    args = parser.parse_args()
    
    # First check if model files already exist
    if check_model_files():
        print("Model files already exist. You can use them with chat_with_model.py")
        if args.train:
            response = input("Do you want to train a new model anyway? (y/n): ")
            if response.lower() != 'y':
                return
    else:
        create_model_dirs()
        print("No model files found")
    
    if args.train:
        # Train a new model
        success = train_new_model(args.message_files, args.epochs, args.augment)
        if success:
            print("\nTraining completed successfully!")
            print("You can now chat with your model using: python chat_with_model.py")
        else:
            print("\nTraining failed or model files not found")
            print("Please check the error messages above")
    else:
        # Provide instructions for manual setup
        print("\nTo train a new model, run one of the following commands:")
        print("  python setup_model.py --train --message_files message_1.json --epochs 5 --augment")
        print("  python clean_and_finetune.py --data_files message_1.json --epochs 5 --augment")
        print("\nOr place your own pre-trained model files in the following locations:")
        print("  fine_tuned_model/model.safetensors")
        print("  checkpoints/best_model/model.safetensors (optional)")
        print("\nAfter setting up the model, you can chat with it using:")
        print("  python chat_with_model.py")

if __name__ == "__main__":
    main()
