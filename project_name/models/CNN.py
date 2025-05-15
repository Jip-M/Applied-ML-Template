import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# --- Step 1: Load Images and Labels from CSVs ---
# Uncomment and adapt this section if you need to load from files

data_dir = "data/cleaned"
csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
images = []
labels = [0] * 52
for _ in range(51):
    labels.append(1)

for file in csv_files:
    img = pd.read_csv(file, header=None).values  # shape: (281, 1000)
    images.append(img)


# --- Step 2: Convert to NumPy Arrays and Preprocess ---
images = np.array(images)  # shape: (29, 281, 1000)
images = images[..., np.newaxis]  # shape: (29, 281, 1000, 1)
# images = images.astype("float32") / 255.0  # Normalize to [0, 1]
labels = np.array(labels).astype(int)  # shape: (29,)

# --- Step 3: Train/Test Split ---
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels)

# --- Step 4: Build the CNN Model ---
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(281, 1000, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')  # 2 classes
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- Step 5: Train the Model ---
model.fit(
    train_images, train_labels,
    epochs=10,
    batch_size=4,  # Small batch size for small dataset
    validation_split=0.2
)

# --- Step 6: Evaluate the Model ---
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")
