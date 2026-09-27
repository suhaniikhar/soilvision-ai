
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model(
    "soil_model.h5",
    compile=False
)

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

prediction = model.predict(
    img_array,
    verbose=0
)

predicted_index = np.argmax(prediction[0])

confidence = float(prediction[0][predicted_index])

# Temporary confidence threshold
THRESHOLD = 0.70

print("--------------------------------")
print("SoilVision AI")
print("--------------------------------")

if confidence < THRESHOLD:

    print("⚠️ It is not a valid image.")
    print("Please upload a soil image.")

else:

    result = classes[predicted_index]

    print("Predicted Soil Type:", result)
    print("Confidence:", round(confidence * 100, 2), "%")

print("--------------------------------")