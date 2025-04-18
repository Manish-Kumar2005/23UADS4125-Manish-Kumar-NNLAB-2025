import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

DATASET_DIR = 'processed_dataset'
IMG_SIZE = 256

def load_images(folder):
    images = []
    for subdir in os.listdir(folder):
        path = os.path.join(folder, subdir)
        for img_file in os.listdir(path):
            img_path = os.path.join(path, img_file)
            img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE)).convert('L')
            img_array = np.array(img) / 255.0  # normalize
            images.append(img_array)
    return np.array(images)

print("Loading images...")
X = load_images(DATASET_DIR)
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print("Loaded:", X.shape)

# Build the autoencoder
input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
x = Flatten()(input_img)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
encoded = Dense(128, activation='relu')(x)

x = Dense(512, activation='relu')(encoded)
x = Dense(1024, activation='relu')(x)
x = Dense(IMG_SIZE * IMG_SIZE, activation='sigmoid')(x)
decoded = Reshape((IMG_SIZE, IMG_SIZE, 1))(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer=Adam(1e-3), loss='binary_crossentropy')
autoencoder.summary()

# Train and store history
print("\nTraining model...")
history = autoencoder.fit(X, X, epochs=15, batch_size=32, shuffle=True, validation_split=0.1)

# Plot training and validation loss
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Binary Crossentropy Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Predict
decoded_imgs = autoencoder.predict(X[:10])

# Visualize original vs reconstructed images
for i in range(5):
    plt.figure(figsize=(6,2))
    plt.subplot(1,2,1)
    plt.imshow(X[i].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(decoded_imgs[i].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    plt.title('Reconstructed')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Evaluate reconstruction error (MSE) and pseudo accuracy
print("\nEvaluating Reconstruction Quality...")
recon_errors = []
for i in range(len(X[:10])):  # Evaluate only on first 10 images
    original = X[i].flatten()
    reconstructed = decoded_imgs[i].flatten()
    mse = mean_squared_error(original, reconstructed)
    recon_errors.append(mse)

avg_mse = np.mean(recon_errors)
print(f"Average MSE (Reconstruction Error): {avg_mse:.6f}")

# Define a pseudo "accuracy" based on error threshold
threshold = 0.01
recon_accuracy = np.mean(np.array(recon_errors) < threshold)
print(f"Pseudo Reconstruction Accuracy: {recon_accuracy * 100:.2f}%")
