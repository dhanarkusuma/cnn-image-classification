import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


# Load dataset dari folder
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "datasets/train", image_size=(150, 150), batch_size=32
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "datasets/test", image_size=(150, 150), batch_size=32
)

# Normalisasi
normalization_layer = layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Model CNN
model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(6, activation="softmax"),
    ]
)

# Compile
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Training
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# Evaluasi
loss, acc = model.evaluate(val_ds)
print("Accuracy:", acc)

# Plot hasil
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.legend()
plt.show()

model.save("model_cnn_intel.h5")
