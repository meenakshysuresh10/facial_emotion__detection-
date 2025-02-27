import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Define dataset paths
train_dir = "dataset/train"
test_dir = "dataset/test"
emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def load_images_from_folder(folder):
    X, y = [], []
    for label, emotion in enumerate(emotions):
        emotion_path = os.path.join(folder, emotion)
        if not os.path.exists(emotion_path): 
            continue
        for img_name in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            img = cv2.resize(img, (48, 48))  # Resize to 48x48
            X.append(img)
            y.append(label)
    return np.array(X), np.array(y)

# Load training and testing images
X_train, y_train = load_images_from_folder(train_dir)
X_test, y_test = load_images_from_folder(test_dir)

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=len(emotions))
y_test = to_categorical(y_test, num_classes=len(emotions))

# Reshape for CNN
X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)

# Save preprocessed data
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("Dataset processing completed and saved successfully.")
