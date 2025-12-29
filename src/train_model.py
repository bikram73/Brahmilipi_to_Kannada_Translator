import os
import numpy as np
import tensorflow as tf
import cv2
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def generate_synthetic_images(num_images_per_class=20, image_size=(64, 64)):
    """
    Generate synthetic grayscale images for Kannada characters
    """
    # Kannada characters to generate
    kannada_characters = ['ಅ', 'ಆ', 'ಇ', 'ಈ', 'ಉ', 'ಕ','ಊ']
    
    images = []
    labels = []
    
    for char in kannada_characters:
        for _ in range(num_images_per_class):
            # Create a blank image
            img = np.zeros(image_size, dtype=np.float32)
            
            # Add random noise
            noise = np.random.normal(0, 0.1, image_size)
            img += noise
            
            # Add a basic shape or pattern to simulate character
            center = image_size[0] // 2
            cv2.circle(img, (center, center), 20, 1.0, -1)
            
            # Simulate some character-like variations
            if char == 'ಅ':
                cv2.line(img, (center-10, center), (center+10, center), 0.5, 2)
            elif char == 'ಆ':
                cv2.rectangle(img, (center-10, center-10), (center+10, center+10), 0.5, 2)
            
            # Normalize
            img = (img - img.min()) / (img.max() - img.min())
            
            images.append(img.reshape(image_size[0], image_size[1], 1))
            labels.append(char)
    
    return np.array(images), np.array(labels)

def train_advanced_model(mapping_file='data/mapping.json'):
    """
    Training process with synthetic data generation
    """
    try:
        # Generate synthetic images
        X, y = generate_synthetic_images()

        # Encode labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(y)
        
        # One-hot encode
        num_classes = len(label_encoder.classes_)
        one_hot_labels = to_categorical(encoded_labels, num_classes=num_classes)

        # Create character mappings
        character_to_int = {label: idx for idx, label in enumerate(label_encoder.classes_)}
        int_to_character = {idx: label for label, idx in character_to_int.items()}

        # Print dataset information
        print("\n--- Synthetic Dataset Information ---")
        print(f"Total Images: {len(X)}")
        print(f"Number of Classes: {num_classes}")
        print("Class Distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"{cls}: {count} images")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, one_hot_labels, test_size=0.2, random_state=42
        )

        # Create model
        model = Sequential([
            Input(shape=X_train[0].shape),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=50,
            validation_split=0.2,
            verbose=1
        )

        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")

        # Save model and mappings
        model.save('kannada_synthetic_character_model.h5')
        
        with open('character_mappings.json', 'w', encoding='utf-8') as f:
            json.dump({
                'character_to_int': character_to_int,
                'int_to_character': int_to_character
            }, f, ensure_ascii=False)

        return model, history

    except Exception as e:
        print(f"An error occurred during model training: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def predict_character(synthetic_image, model_path='kannada_synthetic_character_model.h5', 
                      mappings_path='character_mappings.json'):
    """
    Predict the character for a synthetic image
    """
    # Load model and mappings
    model = tf.keras.models.load_model(model_path)
    
    with open(mappings_path, 'r', encoding='utf-8') as f:
        mappings = json.load(f)
    
    int_to_character = {int(k): v for k, v in mappings['int_to_character'].items()}

    # Preprocess synthetic image
    processed_img = synthetic_image.reshape(1, 64, 64, 1)
    
    # Predict
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    return int_to_character[predicted_class]

if __name__ == '__main__':
    # Train the model with synthetic data
    train_advanced_model()

    # Optional: Test prediction with a synthetic image
    # synthetic_images, _ = generate_synthetic_images()
    # result = predict_character(synthetic_images[0])
    # print(f"Predicted Character: {result}")