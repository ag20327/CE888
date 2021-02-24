import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

image_size = (299, 299) #Expected input of Xception with 3 channels
batch_size = 32
directory = r"C:\Users\andre\OneDrive\Escritorio\Materias Essex\Data Science\Project1\Fire-not-fire\Train" #Location of folder

#Load the data into Python and divide into training and validation
#The data is loaded shuffled to reduce bias
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    color_mode="rgb",#3 Channels
    batch_size=batch_size,
    image_size=image_size, #Image resizing
    shuffle=True,
    seed=9999,
    validation_split=0.15,
    subset="training",
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(

    directory,
    color_mode="rgb",
    batch_size=batch_size,
    image_size=image_size,
    shuffle=True,
    seed=9999,
    validation_split=0.15,
    subset="validation",
)



#Display the first 9 images with their classes
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
plt.show()

#Data augmentation with experimental layers
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)

#Shows the data augmentation effect on the first image applied 9 times
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images) #se aplica a todas las imagenes del batch(1)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
plt.show()

#Augmentate the complete dataset for trainning and validation
augmented_train_ds = train_ds.map(
  lambda x, y: (data_augmentation(x, training=True), y))

augmented_val_ds = val_ds.map(
  lambda x, y: (data_augmentation(x, training=True), y))

#buffered prefetching for performance
augmented_train_ds = augmented_train_ds.prefetch(buffer_size=32)
augmented_val_ds = augmented_val_ds.prefetch(buffer_size=32)

#Display the first 9 images with data augmentation applied to whole dataset
plt.figure(figsize=(10, 10))
for images, labels in augmented_train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
plt.show()








