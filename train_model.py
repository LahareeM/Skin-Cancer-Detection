import zipfile
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

# Step 1: Download dataset manually and unzip it
# Put train and test folders in 'melanoma_cancer_dataset/' directory

train_data_dir = 'melanoma_cancer_dataset/train'
test_data_dir = 'melanoma_cancer_dataset/test'

# Step 2: Load datasets
batch_size = 16
img_size = (224, 224)

train_ds = keras.utils.image_dataset_from_directory(
    train_data_dir,
    label_mode='binary',
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True
)

test_ds = keras.utils.image_dataset_from_directory(
    test_data_dir,
    label_mode='binary',
    batch_size=batch_size,
    image_size=img_size,
    shuffle=False
)

# Step 3: Preprocessing
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Step 4: Build model
img_shape = (img_size[0], img_size[1], 3)
base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')
base_model.trainable = False

model = Sequential([
    base_model,
    BatchNormalization(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Step 5: Train the model
EPOCHS = 10
history = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, verbose=1)

# Step 6: Save the model
model.save('model.h5')

# Step 7: Plot training history 
train_acc = history.history['accuracy'] 
train_loss = history.history['loss'] 
val_acc = history.history['val_accuracy'] 
val_loss = history.history['val_loss'] 
epochs_range = range(EPOCHS) 
plt.figure(figsize=(14, 5)) 
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_loss, label='Training Loss') 
plt.plot(epochs_range, val_loss, label='Validation Loss') 
plt.legend(loc='upper right') 
plt.title('Accuracy') 
 
plt.subplot(1, 2, 2) 
plt.plot(epochs_range, train_acc, label='Training Accuracy') 
plt.plot(epochs_range, val_acc, label='Validation Accuracy') 
plt.legend(loc='lower right') 
plt.title('Loss') 
 
 
plt.show() 