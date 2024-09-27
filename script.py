import os
import numpy as np
import pandas as pd
from PIL import Image

# Path to training folder
training_folder = "dataset/Testing"

# Define folder paths
healthy_folder = os.path.join(training_folder, "Healthy")
early_blight_folder = os.path.join(training_folder, "Early_Blight")
late_blight_folder = os.path.join(training_folder, "Late_Blight")

# Lists to store features (X_train) and labels (Y_train)
X_train = []
Y_train = []

def process_images_from_folder(folder_path, label, resize_dim=(28, 28)):
    """Function to process images from a folder and add their data to X_train and Y_train."""
    for image_file in os.listdir(folder_path):
        img = Image.open(os.path.join(folder_path, image_file)).convert('L')
        img = img.resize(resize_dim)
        img_array = np.array(img).flatten()
        X_train.append(img_array)
        Y_train.append(label)

process_images_from_folder(healthy_folder, label=0)

process_images_from_folder(early_blight_folder, label=1)
process_images_from_folder(late_blight_folder, label=1)

# Convert X_train and Y_train to pandas DataFrames
X_train_df = pd.DataFrame(X_train)
Y_train_df = pd.DataFrame(Y_train)

# Save X_train and Y_train as separate CSV files without headers
X_train_df.to_csv('X_train.csv', index=False, header=False)
Y_train_df.to_csv('Y_train.csv', index=False, header=False)
