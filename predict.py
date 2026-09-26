predict.py

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model("soil_model.h5")

# =========================
# SOIL CLASSES
# =========================

classes = [
    "Alluvial Soil",
    "Arid Soil",
    "Black Soil",
    "Laterite Soil",
    "Mountain Soil",
    "Red Soil",
    "Yellow Soil"
]

# =========================
# TEST IMAGE
# =========================

img_path = "test_images/test.jpg"

img = image.load_img(
    img_path,
    target_size=(128, 128)
)

img_array = image.img_to_array(img)

# Normalize
img_array = img_array / 255.0

# Add batch dimension
img_array = np.expand_dims(
    img_array,
    axis=0
)

# =========================
# PREDICTION
# =========================

prediction = model.predict(img_array)

predicted_index = np.argmax(prediction)

result = classes[predicted_index]

print("--------------------------------")
print("SoilVision AI")
print("--------------------------------")
print("Predicted Soil Type:", result)
print("--------------------------------")