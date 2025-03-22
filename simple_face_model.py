import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

def create_simple_face_model():
    """
    Create a simple CNN model for face recognition as a fallback
    """
    print("Creating a simple face recognition model...")
    
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='linear')  # Face embedding output
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Save the model
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model_path = "models/simple_face_model.h5"
    model.save(model_path)
    
    print(f"Simple model created and saved to {model_path}")
    return model_path

if __name__ == "__main__":
    model_path = create_simple_face_model()
    print(f"Model created at: {model_path}")
    print("You can now run the application with this simple model.") 