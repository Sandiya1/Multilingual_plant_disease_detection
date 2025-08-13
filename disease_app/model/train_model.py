import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split

# Load the npz file and list keys
data = np.load('plant64.npz')
print("Keys inside npz:", data.files)  # See exactly what's inside

# Example keys, adjust based on print output
X_train = data['X_train']  # images for training
y_train = data['y_train']  # labels for training
X_test = data['X_test']    # images for testing
y_test = data['y_test']    # labels for testing

# Map labels to class names (update with your actual classes)
class_names = {
    0: 'Tomato_Late_blight',
    1: 'Tomato_Healthy',
    2: 'Potato_Early_blight',
    # ... add all your classes here
}

# Define output directories
output_dir = 'plant_dataset'
train_dir = os.path.join(output_dir, 'train')
test_dir = os.path.join(output_dir, 'test')

# Create folders for each class
for folder in [train_dir, test_dir]:
    for class_name in class_names.values():
        os.makedirs(os.path.join(folder, class_name), exist_ok=True)

# Helper function to save images
def save_images(images, labels, base_dir):
    for idx, (img_array, label) in enumerate(zip(images, labels)):
        class_name = class_names[label]
        folder = os.path.join(base_dir, class_name)
        if img_array.dtype != 'uint8':
            img_array = (img_array * 255).astype('uint8')
        img = Image.fromarray(img_array)
        img.save(os.path.join(folder, f"{class_name}_{idx}.png"))

# Save training images
save_images(X_train, y_train, train_dir)
# Save test images
save_images(X_test, y_test, test_dir)

print("Saved train and test images successfully!")
