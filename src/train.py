#!/usr/bin/env python3
"""
Training script for rPPG biometric defense CNN model.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import prepare_data, save_preprocessed_data
from src.model import train_model, evaluate_model, plot_training_history, model_summary


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train CNN model for rPPG biometric defense')
    parser.add_argument('--data_path', type=str, default='data/ppg_spec_maps.unknown',
                       help='Path to the PPG data file')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Proportion of data for testing')
    parser.add_argument('--val_size', type=float, default=0.2,
                       help='Proportion of training data for validation')
    parser.add_argument('--augmentation_factor', type=int, default=5,
                       help='Number of augmented samples per original sample')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--model_save_path', type=str, default='models/cnn_classifier.h5',
                       help='Path to save the trained model')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("rPPG Biometric Defense - CNN Training")
    print("=" * 60)
    
    # Step 1: Prepare data
    print("\n1. Preparing data...")
    data_splits = prepare_data(
        data_path=args.data_path,
        test_size=args.test_size,
        val_size=args.val_size,
        augmentation_factor=args.augmentation_factor,
        random_state=args.random_state
    )
    
    if data_splits[0] is None:
        print("Error: Data preparation failed!")
        return
    
    X_train, X_val, X_test, y_train, y_val, y_test = data_splits
    
    # Save preprocessed data
    print("\n2. Saving preprocessed data...")
    save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Step 3: Train model
    print(f"\n3. Training model for {args.epochs} epochs...")
    model, history = train_model(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_save_path=args.model_save_path
    )
    
    # Step 4: Evaluate model
    print("\n4. Evaluating model...")
    evaluation_results = evaluate_model(model, X_test, y_test)
    
    # Step 5: Plot training history
    print("\n5. Plotting training history...")
    plot_training_history(history)
    
    # Step 6: Print model summary
    print("\n6. Model summary:")
    model_summary(model)
    
    # Step 7: Print final results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Model saved to: {args.model_save_path}")
    print(f"Test Accuracy: {evaluation_results['accuracy']:.4f}")
    print(f"Test Precision: {evaluation_results['precision']:.4f}")
    print(f"Test Recall: {evaluation_results['recall']:.4f}")
    print("=" * 60)
    
    # Save training configuration
    config_path = 'results/training_config.txt'
    os.makedirs('results', exist_ok=True)
    with open(config_path, 'w') as f:
        f.write("Training Configuration:\n")
        f.write(f"Data path: {args.data_path}\n")
        f.write(f"Test size: {args.test_size}\n")
        f.write(f"Validation size: {args.val_size}\n")
        f.write(f"Augmentation factor: {args.augmentation_factor}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Random state: {args.random_state}\n")
        f.write(f"Model save path: {args.model_save_path}\n")
    
    print(f"Training configuration saved to: {config_path}")


if __name__ == "__main__":
    main() 