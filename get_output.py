# From that saved unet model get segmeted image and from classification model classify the images
import random

import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2




IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

# TRAIN_PATH = 'stage1_train/'
TRAIN_PATH = 'TrainImages/'

TEST_PATH = 'TestImages/'

train_ids = next(os.walk(TRAIN_PATH))[1]

test_ids = next(os.walk(TEST_PATH))[1]

print("Length is ", len(train_ids))

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)  # (49, 128, 128, 1)

print(Y_train[0])

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.jpg')[:, :, :IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)[:, :, :1]

        mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        print(mask_.shape)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask

X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Resizing Test Images')
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.jpg')[:, :, :IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('done////////')

image_x = random.randint(0, len(train_ids) - 1)
imshow(X_train[image_x])
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()


model = tf.keras.models.load_model('segmentation_model.h5')
idx = random.randint(0, len(X_train))

preds_train = model.predict(X_train[:int(X_train.shape[0] * 0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0] * 0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

if not os.path.exists("Segmented_wo_Train_images"):
    os.mkdir("Segmented_wo_Train_images")

if not os.path.exists("Segmented_wo_Test_images"):
    os.mkdir("Segmented_wo_Test_images")


from tensorflow.keras.preprocessing import image


def get_result(image_name):
    prediction = tf.keras.models.load_model("cnn_model_skin1.h5")
    img = image.load_img(image_name, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    p = np.argmax(prediction.predict(x))
    print(p)
    plt.imshow(img)
    if p == 0:
        plt.title(label="Maligant",
                  fontsize=20,
                  color="Red")
        print("---------------Maligant---------------\n\n\n")
    elif p == 1:
        plt.title(label="Normal",
                  fontsize=20,
                  color="green")
        print("--------------Normal-------------\n\n\n")
    # plt.show()

    os.remove(image_name)


for i in range(len(preds_test_t)):
    print(i)

    imshow(X_train[i])
    plt.show()
    # imshow(np.squeeze(Y_train[i]))
    # plt.show()
    # imshow(np.squeeze(preds_train_t[i]))
    # plt.show()
    imshow(np.squeeze(preds_train_t[i]))
    plt.axis('off')
    segment_image_path = f"Segmented_wo_Test_images/{i}_seg_test_000{i}.png"
    plt.savefig(segment_image_path)
    get_result(segment_image_path)

for i in range(len(preds_train_t)):
    print(i)
    imshow(X_train[i])
    plt.show()
    # imshow(np.squeeze(Y_train[i]))
    # plt.show()
    # imshow(np.squeeze(preds_train_t[i]))q
    # plt.show()
    imshow(np.squeeze(preds_train_t[i]))
    plt.axis('off')
    segment_image_path = f"Segmented_wo_Train_images/{i}_seg_train_000{i}.jpg"
    plt.savefig(segment_image_path)
    get_result(segment_image_path)











