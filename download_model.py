import os
import requests
import zipfile
import io
import gdown
import shutil

def download_facenet_model():
    """
    Download a pre-trained FaceNet model for facial recognition
    """
    print("Downloading FaceNet Keras model...")
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Option 1: Google Drive link to a FaceNet model
    model_url = "https://drive.google.com/uc?id=1PZ_6Zsy1Vb0s0JmjEmVd8FS99zoMCiN1"
    
    try:
        output_path = "models/facenet_keras.h5"
        gdown.download(model_url, output_path, quiet=False)
        print(f"Model downloaded successfully to {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading from Google Drive: {e}")
        
        # Option 2: Try another source if Google Drive fails
        try:
            print("Trying alternative download source...")
            alt_url = "https://github.com/nyoki-mtl/keras-facenet/raw/master/model/facenet_keras.h5"
            response = requests.get(alt_url, stream=True)
            
            if response.status_code == 200:
                with open("models/facenet_keras.h5", 'wb') as f:
                    shutil.copyfileobj(response.raw, f)
                print("Model downloaded successfully from alternative source")
                return True
            else:
                print(f"Failed to download model: HTTP status {response.status_code}")
                return False
        except Exception as e:
            print(f"Error with alternative download: {e}")
            return False

if __name__ == "__main__":
    # Check if model already exists
    if os.path.exists("models/facenet_keras.h5"):
        print("FaceNet model already exists. If you want to re-download, please delete the existing file first.")
    else:
        success = download_facenet_model()
        
        if success:
            print("\nSetup complete! You can now run the application with: python app.py")
        else:
            print("\nFailed to download model automatically.")
            print("Please download a FaceNet Keras model manually and place it in the 'models' directory as 'facenet_keras.h5'")
            print("You can find models at: https://github.com/nyoki-mtl/keras-facenet or https://www.kaggle.com/datasets/suicaokhoailang/facenet-keras") 