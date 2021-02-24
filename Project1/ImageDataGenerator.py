from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

image_size = (299, 299) #Expected input of Xception with 3 channels
batch_size = 32
directory = r"C:\Users\andre\OneDrive\Escritorio\Materias Essex\Data Science\Project1\Fire-not-fire\Train" #Location of folder

train_datagen = ImageDataGenerator(
    #rescale=1./255
    brightness_range=[1,5],
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)

val_datagen = ImageDataGenerator(
    #rescale=1./255
    brightness_range=[0.3,0.9],
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_directory(
    directory,
    color_mode="rgb",
    batch_size=batch_size,
    target_size=image_size,
    shuffle=True,
    seed=9999,
    subset="training",
)

validation_generator = val_datagen.flow_from_directory(
    directory,
    color_mode="rgb",
    batch_size=batch_size,
    target_size=image_size,
    shuffle=True,
    seed=9999,
    subset="validation",
)

chunk = train_generator.next()
ch=chunk[0]


#Display the first 9 images with data augmentation applied to whole dataset with DataGenerator
plt.figure(figsize=(10, 10))
for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(ch[i].astype("uint8"))
        plt.axis("off")
plt.show()

#https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator