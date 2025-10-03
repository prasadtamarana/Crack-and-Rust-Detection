# Imprt Libraries
import os
import numpy as np
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Dataset Paths
base_dir = r"C:\Users\prasa\OneDrive\Desktop\crack_dataset_2"
train_dir = r"C:\Users\prasa\OneDrive\Desktop\crack_dataset_2\train"
val_dir = r"C:\Users\prasa\OneDrive\Desktop\crack_dataset_2\valid"

# ImageDataGenerators with augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load datasets
train_ds = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

val_ds = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Class weight calculation
labels = train_ds.classes
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weight_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weight_dict)

# CNN model Building
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001),  
              loss='binary_crossentropy',           
              metrics=['accuracy'])

# Train model
model.fit(
    train_ds,
    epochs=30,  
    validation_data=val_ds,
    class_weight=class_weight_dict
)

# Save model
model.save(os.path.join(base_dir, "rust_vs_crust_model.h5"))
print("Model training complete and saved.")



