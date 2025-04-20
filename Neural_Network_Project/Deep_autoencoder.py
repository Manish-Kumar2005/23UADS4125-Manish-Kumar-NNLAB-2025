import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Constants
DATASET_DIR = 'processed_dataset'
IMG_SIZE = 256

# Load images
def load_images(folder):
    images = []
    for subdir in os.listdir(folder):
        path = os.path.join(folder, subdir)
        for img_file in os.listdir(path):
            img_path = os.path.join(path, img_file)
            img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE)).convert('L')
            img_array = np.array(img) / 255.0  # Normalize
            images.append(img_array)
    return np.array(images)

# Load dataset
print("Loading images...")
X = load_images(DATASET_DIR)
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print("Total images loaded:", X.shape)

# Train-test split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# Build Convolutional Autoencoder
input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 1))

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Compile model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer=Adam(1e-3), loss='mse')
autoencoder.summary()

# Train model
print("\nTraining model...")
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = autoencoder.fit(
    X_train, X_train,
    epochs=15,
    batch_size=32,
    shuffle=True,
    validation_split=0.1,
    callbacks=[early_stop]
)

# Plot training history
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Predict on train and test
print("\nGenerating reconstructions...")
decoded_train = autoencoder.predict(X_train)
decoded_test = autoencoder.predict(X_test)

# Clamp values for safe viewing
decoded_train = np.clip(decoded_train, 0, 1)
decoded_test = np.clip(decoded_test, 0, 1)

# Show original vs reconstructed (from test set)
for i in range(5):
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    axs[0].imshow(X_test[i].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    axs[0].set_title('Original')
    axs[0].axis('off')

    axs[1].imshow(decoded_test[i].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    axs[1].set_title('Reconstructed')
    axs[1].axis('off')
    plt.tight_layout()
    plt.show()

# Evaluation Function
def evaluate_reconstruction(true_imgs, reconstructed_imgs):
    errors = [mean_squared_error(true_imgs[i].flatten(), reconstructed_imgs[i].flatten()) for i in range(len(true_imgs))]
    avg_mse = np.mean(errors)

    # Pseudo accuracy based on 90th percentile threshold
    threshold = np.percentile(errors, 90)
    pseudo_acc = np.mean(np.array(errors) < threshold) * 100

    return avg_mse, pseudo_acc

# Train Accuracy
print("\nEvaluating Train Accuracy...")
train_mse, train_acc = evaluate_reconstruction(X_train, decoded_train)
print(f"Train MSE: {train_mse:.6f}")
print(f"Train Pseudo Accuracy: {train_acc:.2f}%")

# Test Accuracy
print("\nEvaluating Test Accuracy...")
test_mse, test_acc = evaluate_reconstruction(X_test, decoded_test)
print(f"Test MSE: {test_mse:.6f}")
print(f"Test Pseudo Accuracy: {test_acc:.2f}%")
