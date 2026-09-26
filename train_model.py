import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =========================
# DATASET PATH
# =========================

train_dir = "dataset/train"

# =========================
# IMAGE PREPROCESSING
# =========================

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# =========================
# CNN MODEL
# =========================

model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(128, 128, 3)
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        7,
        activation="softmax"
    )
])

# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# TRAIN MODEL
# =========================

model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# =========================
# SAVE MODEL
# =========================

model.save("soil_model.h5")

print("Model training completed.")
print("Model saved as soil_model.h5")