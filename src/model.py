"""
CNN model for rPPG biometric defense project.
Contains the model architecture and training utilities.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os


def create_cnn_model(input_shape=(10, 31, 1), dropout_rate=0.5):
    """Create the CNN model for PPG spectral map classification."""
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),
        
        # Flatten and dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.7),  # Higher dropout for dense layer
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model


def create_callbacks(model_save_path='models/cnn_classifier.h5', patience=20):
    """Create training callbacks for better model training."""
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    callbacks = [
        # Save best model
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when plateau is reached
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks


def train_model(X_train, y_train, X_val, y_val, epochs=200, batch_size=32, 
                model_save_path='models/cnn_classifier.h5'):
    """Train the CNN model."""
    # Create model
    model = create_cnn_model()
    
    # Create callbacks
    callbacks = create_callbacks(model_save_path)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history


def evaluate_model(model, X_test, y_test, save_results=True):
    """Evaluate the trained model on test data."""
    # Make predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate metrics
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save results if requested
    if save_results:
        os.makedirs('results', exist_ok=True)
        with open('results/evaluation_results.txt', 'w') as f:
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
            f.write(f"Test Precision: {test_precision:.4f}\n")
            f.write(f"Test Recall: {test_recall:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_test, y_pred))
    
    return {
        'loss': test_loss,
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_pred_prob
    }


def plot_training_history(history, save_plots=True):
    """Plot training history (accuracy and loss curves)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_plots:
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
    
    plt.show()


def load_trained_model(model_path='models/cnn_classifier.h5'):
    """Load a trained model from file."""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def model_summary(model):
    """Print model summary and save it to file."""
    # Print to console
    model.summary()
    
    # Save to file
    os.makedirs('results', exist_ok=True)
    with open('results/model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))


if __name__ == "__main__":
    # Example usage
    print("Creating CNN model...")
    model = create_cnn_model()
    model_summary(model)
    
    print("\nModel created successfully!")
    print("Use train_model() function to train the model with your data.") 