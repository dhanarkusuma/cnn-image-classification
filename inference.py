import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model("model_cnn_intel.h5")

# Label sesuai dataset Intel
class_names = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# Load gambar
img_path = "datasets/inference/64.jpg"  # ganti dengan gambar kamu
img = image.load_img(img_path, target_size=(150, 150))

# Convert ke array
img_array = image.img_to_array(img)

# Normalisasi (WAJIB sama seperti training)
img_array = img_array / 255.0

# Tambah dimensi batch
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)

# Ambil hasil
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions)

print("Predicted:", predicted_class)
print("Confidence:", confidence)
