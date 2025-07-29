"""
Data preprocessing for rPPG biometric defense project.
Handles data loading, augmentation, and preparation for training.
"""

import pickle as pk
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os


def load_ppg_data(data_path='data/ppg_spec_maps.unknown'):
    """Load PPG spectral maps from pickle file."""
    try:
        X, y = pk.load(open(data_path, 'rb'))
        print(f"Loaded data: X shape {X.shape}, y shape {y.shape}")
        return X, y
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def visualize_sample_data(X, num_samples=10):
    """Visualize sample PPG maps."""
    plt.figure(figsize=(15, 6))
    for i in range(min(num_samples, len(X))):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X[i], cmap='viridis')
        plt.title(f'Sample {i+1}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def augment_data(X_train, y_train, augmentation_factor=5):
    """Augment training data using various transformations."""
    # Configure data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
    )
    
    datagen.fit(X_train)
    
    # Generate augmented data
    augmented_images = []
    augmented_labels = []
    
    for i in range(len(X_train)):
        img = X_train[i].reshape((1, *X_train[i].shape))
        label = y_train[i]
        counter = 0
        
        for batch in datagen.flow(img, batch_size=1):
            augmented_images.append(batch[0])
            augmented_labels.append(label)
            counter += 1
            if counter >= augmentation_factor:
                break
    
    # Combine original and augmented data
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)
    
    X_train_augmented = np.concatenate((X_train, augmented_images), axis=0)
    y_train_augmented = np.concatenate((y_train, augmented_labels), axis=0)
    
    print(f"Original training samples: {len(X_train)}")
    print(f"Augmented training samples: {len(X_train_augmented)}")
    
    return X_train_augmented, y_train_augmented


def prepare_data(data_path='data/ppg_spec_maps.unknown', test_size=0.2, val_size=0.2, 
                augmentation_factor=5, random_state=42):
    """Complete data preparation pipeline."""
    # Load data
    X, y = load_ppg_data(data_path)
    if X is None:
        return None, None, None, None, None, None
    
    # Visualize sample data
    visualize_sample_data(X)
    
    # Print data statistics
    print(f"Data shape: {X.shape}")
    print(f"Data max value: {X.max()}")
    print(f"Data min value: {X.min()}")
    print(f"Label distribution: {np.bincount(y)}")
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Augment training data
    X_train_augmented, y_train_augmented = augment_data(
        X_train, y_train, augmentation_factor
    )
    
    # Split augmented training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_augmented, y_train_augmented, 
        test_size=val_size, random_state=random_state
    )
    
    print(f"Final data splits:")
    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Testing: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test, 
                         output_dir='data/preprocessed'):
    """Save preprocessed data to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(f'{output_dir}/X_train.npy', X_train)
    np.save(f'{output_dir}/X_val.npy', X_val)
    np.save(f'{output_dir}/X_test.npy', X_test)
    np.save(f'{output_dir}/y_train.npy', y_train)
    np.save(f'{output_dir}/y_val.npy', y_val)
    np.save(f'{output_dir}/y_test.npy', y_test)
    
    print(f"Preprocessed data saved to {output_dir}")


def load_preprocessed_data(data_dir='data/preprocessed'):
    """Load preprocessed data from files."""
    try:
        X_train = np.load(f'{data_dir}/X_train.npy')
        X_val = np.load(f'{data_dir}/X_val.npy')
        X_test = np.load(f'{data_dir}/X_test.npy')
        y_train = np.load(f'{data_dir}/y_train.npy')
        y_val = np.load(f'{data_dir}/y_val.npy')
        y_test = np.load(f'{data_dir}/y_test.npy')
        
        print("Preprocessed data loaded successfully")
        return X_train, X_val, X_test, y_train, y_val, y_test
    except FileNotFoundError:
        print(f"Preprocessed data not found in {data_dir}")
        return None, None, None, None, None, None


if __name__ == "__main__":
    # Example usage
    print("Preparing data...")
    data_splits = prepare_data()
    
    if data_splits[0] is not None:
        X_train, X_val, X_test, y_train, y_val, y_test = data_splits
        save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test)
        print("Data preparation completed successfully!")
    else:
        print("Data preparation failed!") 