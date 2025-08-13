import numpy as np
from sklearn.model_selection import train_test_split

# Load data
data = np.load('plant64.npz')
X_train_full, y_train_full = data['train_images'], data['train_labels']
X_test, y_test = data['test_images'], data['test_labels']

# Downsample to ~10K
X_small, _, y_small, _ = train_test_split(
    X_train_full, y_train_full,
    train_size=10000, stratify=y_train_full, random_state=42
)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_small, y_small, test_size=0.2, stratify=y_small, random_state=42
)

# Normalize
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0
