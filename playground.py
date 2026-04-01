import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


# Load dataset dari folder
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "datasets/train", image_size=(150, 150), batch_size=32
)
for x, y in train_ds.take(1000):
    print("X shape:", x.shape)
    print("Y:", y)
    print("Label pertama:", y[0].numpy())
