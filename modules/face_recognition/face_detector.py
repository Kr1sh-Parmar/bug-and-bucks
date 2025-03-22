import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

class FaceRecognition:
    def __init__(self, model_path=None):
        # Initialize OpenCV face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load face embedding model (alternative to dlib)
        if model_path and os.path.exists(model_path):
            self.embedding_model = load_model(model_path)
        else:
            self.embedding_model = None
            print("Warning: No embedding model provided. Using basic face detection only.")
        
        self.face_database = {}  # Will store user face embeddings
    
    def detect_face(self, image):
        """Detect faces in an image and return the face regions"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        face_images = []
        for (x, y, w, h) in faces:
            face_img = image[y:y+h, x:x+w]
            face_images.append(face_img)
            
        return face_images, faces
    
    def get_face_embedding(self, face_image):
        """Generate embedding vector for a face image"""
        if self.embedding_model is None:
            raise ValueError("No embedding model loaded")
        
        # Preprocess the image for the model
        face_image = cv2.resize(face_image, (160, 160))
        face_image = face_image.astype(np.float32) / 255.0
        face_image = np.expand_dims(face_image, axis=0)
        
        # Get embedding
        embedding = self.embedding_model.predict(face_image)[0]
        return embedding
    
    def register_face(self, user_id, image):
        """Register a user's face in the database"""
        face_images, faces = self.detect_face(image)
        
        if not face_images:
            return False, "No face detected"
        
        if len(face_images) > 1:
            return False, "Multiple faces detected"
        
        if self.embedding_model is None:
            # Fallback mode when no embedding model is available
            # Store the face image directly (less secure but allows basic functionality)
            # In a real system, this would not be recommended
            self.face_database[user_id] = {
                'face_image': cv2.resize(face_images[0], (100, 100)),
                'is_simple_mode': True
            }
            return True, "Face registered in simple mode (no embedding model available)"
        
        embedding = self.get_face_embedding(face_images[0])
        self.face_database[user_id] = embedding
        
        return True, "Face registered successfully"
    
    def verify_face(self, user_id, image, threshold=0.6):
        """Verify if the face in the image matches the registered user"""
        if user_id not in self.face_database:
            return False, "User not registered"
        
        face_images, _ = self.detect_face(image)
        
        if not face_images:
            return False, "No face detected"
        
        if len(face_images) > 1:
            return False, "Multiple faces detected"
        
        # Check if we're in simple mode (no embedding model)
        if self.embedding_model is None or isinstance(self.face_database[user_id], dict) and self.face_database[user_id].get('is_simple_mode'):
            # Basic fallback verification using template matching
            # This is less secure but allows testing
            stored_face = self.face_database[user_id].get('face_image')
            if stored_face is None:
                return False, "Stored face not found in simple mode"
            
            # Resize the current face to match stored face
            current_face = cv2.resize(face_images[0], (100, 100))
            
            # Convert to grayscale for template matching
            stored_gray = cv2.cvtColor(stored_face, cv2.COLOR_BGR2GRAY)
            current_gray = cv2.cvtColor(current_face, cv2.COLOR_BGR2GRAY)
            
            # Use template matching as a simple comparison
            # In a real system, you would use a more robust method
            similarity = cv2.matchTemplate(current_gray, stored_gray, cv2.TM_CCOEFF_NORMED)[0][0]
            
            if similarity > 0.5:  # Arbitrary threshold
                return True, f"Face verified with basic similarity {similarity:.2f}"
            else:
                return False, f"Basic verification failed (similarity: {similarity:.2f})"
        
        # Regular verification using embeddings
        embedding = self.get_face_embedding(face_images[0])
        registered_embedding = self.face_database[user_id]
        
        # Calculate cosine similarity
        similarity = np.dot(embedding, registered_embedding)
        similarity /= (np.linalg.norm(embedding) * np.linalg.norm(registered_embedding))
        
        if similarity > threshold:
            return True, f"Face verified with confidence {similarity:.2f}"
        else:
            return False, f"Verification failed (similarity: {similarity:.2f})" 