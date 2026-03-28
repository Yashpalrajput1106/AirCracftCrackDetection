
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

# Load images function
def load_images(folder, label, img_size=256):
    images = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        img = cv2.imread(file_path)
        if img is None:
            print(f"Skipping {filename}: Unable to load.")
            continue
        img = cv2.resize(img, (img_size, img_size))
        images.append((img, label))
    return images

# Load dataset
uncracked_images = load_images("C:\\Users\\vedha\\OneDrive\\Desktop\\miniproject", label=0)
cracked_images = load_images("C:\\Users\\vedha\\OneDrive\\Desktop\\miniproject", label=1)

# Combine data
data = uncracked_images + cracked_images
np.random.shuffle(data)

# Split into images and labels
images, labels = zip(*data)
images = np.array(images) / 255.0  # Normalize
labels = np.array(labels)

# Train-test split
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42, shuffle=True)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Predictions
train_pred = model.predict(train_images)
test_pred = model.predict(test_images)
train_pred_labels = (train_pred > 0.5).astype(int)
test_pred_labels = (test_pred > 0.5).astype(int)

# Confusion matrix
train_cm = confusion_matrix(train_labels, train_pred_labels)
test_cm = confusion_matrix(test_labels, test_pred_labels)

# Plot confusion matrix
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(train_cm, annot=True, fmt='d', ax=ax[0], cmap='Blues')
ax[0].set_title("Train Confusion Matrix")
sns.heatmap(test_cm, annot=True, fmt='d', ax=ax[1], cmap='Reds')
ax[1].set_title("Test Confusion Matrix")
plt.show()

# ROC Curve
y_scores = model.predict(test_images).ravel()
fpr, tpr, _ = roc_curve(test_labels, y_scores)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# Display some test predictions
fig, axs = plt.subplots(1, min(4, len(test_images)), figsize=(12, 4))
for i in range(min(4, len(test_images))):
    axs[i].imshow(test_images[i])
    axs[i].axis('off')
    axs[i].set_title("Cracked" if test_pred_labels[i] else "Uncracked")
plt.show()
