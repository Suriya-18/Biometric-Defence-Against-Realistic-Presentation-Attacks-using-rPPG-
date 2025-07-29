#!/usr/bin/env python3
"""
Inference script for rPPG biometric defense CNN model.
"""

import argparse
import os
import sys
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import load_trained_model


def load_test_data(data_path):
    """Load test data for inference."""
    try:
        if data_path.endswith('.pkl') or data_path.endswith('.unknown'):
            # Load pickle file
            data = pk.load(open(data_path, 'rb'))
            if isinstance(data, tuple):
                X, y = data
                return X, y
            else:
                return data, None
        elif data_path.endswith('.npy'):
            # Load numpy array
            data = np.load(data_path)
            return data, None
        else:
            print(f"Unsupported file format: {data_path}")
            return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def predict_attack(model, X, threshold=0.5):
    """Make predictions using the trained model."""
    # Make predictions
    probabilities = model.predict(X)
    predictions = (probabilities > threshold).astype(int)
    
    return predictions, probabilities


def visualize_predictions(X, predictions, probabilities, num_samples=10):
    """Visualize predictions with confidence scores."""
    num_samples = min(num_samples, len(X))
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(num_samples):
        axes[i].imshow(X[i], cmap='viridis')
        pred_class = "Attack" if predictions[i] == 1 else "Genuine"
        confidence = probabilities[i][0] if predictions[i] == 1 else 1 - probabilities[i][0]
        axes[i].set_title(f'{pred_class}\nConf: {confidence:.3f}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def print_prediction_summary(predictions, probabilities, true_labels=None):
    """Print a summary of predictions."""
    print("\n" + "=" * 50)
    print("PREDICTION SUMMARY")
    print("=" * 50)
    
    # Count predictions
    genuine_count = np.sum(predictions == 0)
    attack_count = np.sum(predictions == 1)
    total_count = len(predictions)
    
    print(f"Total samples: {total_count}")
    print(f"Predicted Genuine: {genuine_count} ({genuine_count/total_count*100:.1f}%)")
    print(f"Predicted Attack: {attack_count} ({attack_count/total_count*100:.1f}%)")
    
    # Average confidence
    avg_confidence = np.mean(np.maximum(probabilities, 1 - probabilities))
    print(f"Average confidence: {avg_confidence:.3f}")
    
    # If true labels are available, calculate accuracy
    if true_labels is not None:
        accuracy = np.mean(predictions == true_labels)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(true_labels, predictions)
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions))
    
    print("=" * 50)


def save_predictions(predictions, probabilities, output_path='results/predictions.txt'):
    """Save predictions to a file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("Sample_ID,Predicted_Class,Probability,Confidence\n")
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            pred_class = "Attack" if pred == 1 else "Genuine"
            confidence = prob[0] if pred == 1 else 1 - prob[0]
            f.write(f"{i},{pred_class},{prob[0]:.4f},{confidence:.4f}\n")
    
    print(f"Predictions saved to: {output_path}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Make predictions with trained rPPG model')
    parser.add_argument('--model_path', type=str, default='models/cnn_classifier.h5',
                       help='Path to the trained model')
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to the input data file')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize predictions')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predictions to file')
    parser.add_argument('--output_path', type=str, default='results/predictions.txt',
                       help='Path to save predictions')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("rPPG Biometric Defense - Inference")
    print("=" * 60)
    
    # Step 1: Load model
    print(f"\n1. Loading model from {args.model_path}...")
    model = load_trained_model(args.model_path)
    if model is None:
        print("Error: Could not load model!")
        return
    
    # Step 2: Load data
    print(f"\n2. Loading data from {args.input_path}...")
    X, y_true = load_test_data(args.input_path)
    if X is None:
        print("Error: Could not load data!")
        return
    
    print(f"Loaded data shape: {X.shape}")
    
    # Step 3: Make predictions
    print(f"\n3. Making predictions with threshold {args.threshold}...")
    predictions, probabilities = predict_attack(model, X, args.threshold)
    
    # Step 4: Print summary
    print_prediction_summary(predictions, probabilities, y_true)
    
    # Step 5: Visualize (if requested)
    if args.visualize:
        print("\n4. Visualizing predictions...")
        visualize_predictions(X, predictions, probabilities)
    
    # Step 6: Save predictions (if requested)
    if args.save_predictions:
        print("\n5. Saving predictions...")
        save_predictions(predictions, probabilities, args.output_path)
    
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main() 