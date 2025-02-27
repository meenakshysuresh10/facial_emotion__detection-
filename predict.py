import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained emotion detection model
MODEL_PATH = r"C:\Users\menak\facial_emotion_detection\emotion_model.h5"
model = load_model(MODEL_PATH)

# Define the class labels for emotion prediction
CLASS_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def predict_emotion(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Check if the image is loaded successfully
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Convert image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize image to match model input size (48x48)
    img = cv2.resize(img, (48, 48)) / 255.0

    # Expand dimensions to match model input shape (1, 48, 48, 1)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)

    # Predict emotion
    predictions = model.predict(img)
    predicted_label = CLASS_LABELS[np.argmax(predictions)]

    print(f"Predicted Emotion: {predicted_label}")

# Test with the correct image path
test_image_path = r"C:\Users\menak\facial_emotion_detection\dataset\test\surprise\PrivateTest_642696.jpg"
predict_emotion(test_image_path)
