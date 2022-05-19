# For making model which is used for getting output of classification
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from pylab import imread, subplot, imshow, show
import cv2
import os
import os

train_path = 'Segmented_Train_images'
train_m_path = os.path.join(train_path,'Maligant')
train_n_path = os.path.join(train_path,'Normal')

print(train_path)
print(train_m_path)
print(train_n_path)

test_path = 'Segmented_Test_images'
test_m_path = os.path.join(train_path,'Maligant')
test_n_path = os.path.join(train_path,'Normal')

print(test_path)
print(test_m_path)
print(test_n_path)

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    validation_split=0.25,
    image_size=(224,224),
    batch_size=35,
    subset='training',
    seed=50 )

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    validation_split=0.25,
    image_size=(224,224),
    batch_size=35,
    subset='validation',
    seed=50
    )

test_data=tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    image_size=(224,224),
    batch_size=35,
    seed=50
    )

class_names = ['Maligant', 'Normal']

print(class_names)

train_data.class_names = class_names
val_data.class_names = class_names

# plt.figure(figsize=(14, 14))
for images, labels in train_data.take(1):
    for i in range(20):
        # ax = plt.subplot(3, 3, i + 1)
        imshow(images[i].numpy().astype("uint8"))
        plt.title(train_data.class_names[labels[i]])
        plt.axis("off")
        plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,Input
from tensorflow.keras.layers import Dense

model=Sequential()

#CNN Layer
model.add(Conv2D(32,(3,3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(50,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(200,activation='relu'))
model.add(Dense(2, activation='softmax'))

model.summary()
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

samples=2637
batch_size=35
new_values= samples//batch_size
value=print(new_values)
history = model.fit(
          train_data,
          steps_per_epoch = value,
          epochs=100,
          validation_data=val_data,
                             )

model.evaluate(val_data)
model.evaluate(test_data)
model.save("cnn_model_skin1.h5")

plt.plot(history.history['accuracy'])
plt.title('Train model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

plt.plot(history.history['loss'])
plt.title('Train Model loss')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

loss_train = history.history['loss']
loss_val = history.history['val_loss']
plt.plot(loss_train, 'g', label='Training loss')
plt.plot(loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

accuracy_train = history.history['accuracy']
accuracy_val = history.history['val_accuracy']
plt.plot(accuracy_train, 'g', label='Training accuracy')
plt.plot(accuracy_val, 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
