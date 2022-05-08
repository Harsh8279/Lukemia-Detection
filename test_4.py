import random

import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt


IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

# TRAIN_PATH = 'stage1_train/'
TRAIN_PATH = 'TrainImages/'

TEST_PATH = 'TestImages/'

'''

TrainImages/
====================================
['Im063_1', 'Im004_1', 'Im018_1', 'Im049_1', 'Im060_1', 'Im026_1', 'Im058_1', 'Im024_1', 'Im051_1', 'Im029_1', 'Im028_1', 'Im027_1', 'Im053_1', 'Im033_1', 'Im059_1', 'Im002_1', 'Im015_1', 'Im016_1', 'Im003_1', 'Im001_1', 'Im006_1', 'Im008_1', 'Im056_1', 'Im052_1', 'Im048_1', 'Im020_1', 'Im005_1', 'Im019_1', 'Im022_1', 'Im054_1', 'Im025_1', 'Im057_1', 'Im013_1', 'Im009_1', 'Im012_1', 'Im050_1', 'Im062_1', 'Im055_1', 'Im017_1', 'Im031_1', 'Im007_1', 'Im011_1', 'Im010_1', 'Im021_1', 'Im061_1', 'Im023_1', 'Im014_1', 'Im032_1', 'Im030_1']
====================================
[]
====================================

'''

train_ids = next(os.walk(TRAIN_PATH))[1]        # we want folder name of Images so we have taken the first element of next(os.walk(TRAIN_PATH))

test_ids = next(os.walk(TEST_PATH))[1]

print("Length is ",len(train_ids))

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

# print(X_train.shape)        # (49, 128, 128, 3)

# print(len(X_train[0][0][0]))

'''

[
[
    [
        [0 0 0] * 128  ==> shape - (128,3)
        
    ]* 128          ==> shape - (128,128,3)
]*49  ==> shape - (49,128,128,3)
]

'''

Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)       # (49, 128, 128, 1)

print(Y_train[0])

'''

[
[

    [False False False] * 128
] * 128

]*49

'''

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.jpg')[:, :, :IMG_CHANNELS]
    # # img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]

    # print("Before Shape of Image ",img.shape)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    # print("After Shape of Image ",img.shape)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)
    # print(next(os.walk(path+'/masks/')))        # ('TrainImages/Im063_1/masks/', [], ['Im063_1_(676,576.png', 'Im063_1_(1336,1236.png', 'Im063_1_(1240,1140.png', 'Im063_1_(1132,1032.png', 'Im063_1_(700,600.png', 'Im063_1_(1708,1608.png', 'Im063_1_(799,699.png', 'Im063_1_(1000,900.png', 'Im063_1_(670,570.png', 'Im063_1_(1381,1281.png', 'Im063_1_(1318,1218.png', 'Im063_1_(1237,1137.png', 'Im063_1_(1144,1044.png', 'Im063_1_(1018,918.png', 'Im063_1_(1780,1680.png', 'Im063_1_(985,885.png', 'Im063_1_(754,654.png', 'Im063_1_(604,504.png', 'Im063_1_(817,717.png'])
    for mask_file in next(os.walk(path+'/masks/'))[2]:
        mask_ = imread(path+'/masks/'+mask_file)[:, :, :1]
        # mask_ = imread(path+'/masks/'+mask_file)

        mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        print(mask_.shape)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask
# print(X_train[0])

'''
for i in Y_train:
    print("______________________________________________________________")
    print(i)
    print("______________________________________________________________")

'''

X_test = np.zeros((len(test_ids),IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS),dtype=np.uint8)
sizes_test = []
print('Resizing Test Images')
for n,id_ in tqdm(enumerate(test_ids),total=len(test_ids)):
    path = TEST_PATH+id_
    img = imread(path+'/images/'+id_+'.jpg')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0],img.shape[1]])
    img = resize(img,(IMG_HEIGHT,IMG_WIDTH),mode='constant', preserve_range=True)
    X_test[n] = img

print('done////////')

image_x = random.randint(0,len(train_ids)-1)
imshow(X_train[image_x])
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()


# Build model

inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
# print(inputs)

s = tf.keras.layers.Lambda(lambda x: x/255)(inputs)

c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(c1)

p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)
# print(c1)


c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(c2)

p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(c3)

p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(c4)

p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.2)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(c5)


# Expansive Path

u6 = tf.keras.layers.Conv2DTranspose(128,(2,2),strides=(2,2),padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6,c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64,(2,2),strides=(2,2),padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7,c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(c7)


u8 = tf.keras.layers.Conv2DTranspose(32,(2,2),strides=(2,2),padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8,c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(c8)


u9 = tf.keras.layers.Conv2DTranspose(16,(2,2),strides=(2,2),padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9,c1])
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(c9)


outputs = tf.keras.layers.Conv2D(1,(1,1),activation='sigmoid')(c9)

model = tf.keras.Model(inputs=[inputs],outputs=[outputs])

model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5',verbose=1,save_best_only=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2,monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs')
]

# results = model.fit(X_train,Y_train, validation_split=0.1, batch_size=16, epochs = 25,callbacks=callbacks)
results = model.fit(X_train,Y_train, validation_split=0.1, batch_size=16, epochs = 500)
model.save('segmentation_model.h5')
print(results.history.keys())

plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
##################################################################################################################################
import pickle
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

idx = random.randint(0,len(X_train))

preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)],verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):],verbose=1)
preds_test = model.predict(X_test,verbose=1)

preds_train_t = (preds_train>0.5).astype(np.uint8)
preds_val_t = (preds_val>0.5).astype(np.uint8)
preds_test_t = (preds_test>0.5).astype(np.uint8)
import cv2
if not os.path.exists("Segmented_Train_images"):
    os.mkdir("Segmented_Train_images")

if not os.path.exists("Segmented_Test_images"):
    os.mkdir("Segmented_Test_images")
# perform a sanity check on some random training samples
for i in range(len(preds_test_t)):
    print(i)

    # ix = random.randint(0,len(preds_train_t))
    imshow(X_train[i])
    plt.show()
    imshow(np.squeeze(Y_train[i]))
    plt.show()
    # np.squeeze(preds_train_t[i])
    imshow(np.squeeze(preds_train_t[i]))
    plt.show()
    imshow(np.squeeze(preds_train_t[i]))
    plt.axis('off')
    # plt.axis('off')
    segment_image_path = f"Segmented_Test_images/{i}_seg_test_000{i}.png"
    plt.savefig(segment_image_path)
    # plt.savefig('abc.jpg')
    # plt.savefig('foo.jpg', bbox_inches='tight')
    # from PIL import Image
    #
    # im = Image.fromarray(np.squeeze(preds_train_t[i]))
    # segment_image_path = f"Segmented_Test_images/{i}_seg_test_000{i}.jpg"
    # im.save(segment_image_path)
    # segment_image_path = f"Segmented_Test_images/{i}_seg_test_000{i}.jpg"
    # x = cv2.imwrite(segment_image_path, )


for i in range(len(preds_train_t)):
    print(i)

    # ix = random.randint(0,len(preds_train_t))
    imshow(X_train[i])
    plt.show()
    imshow(np.squeeze(Y_train[i]))
    plt.show()
    imshow(np.squeeze(preds_train_t[i]))
    plt.show()
    imshow(np.squeeze(preds_train_t[i]))
    plt.axis('off')
    segment_image_path = f"Segmented_Train_images/{i}_seg_train_000{i}.jpg"
    plt.savefig(segment_image_path)
    # x = cv2.imwrite(segment_image_path, np.squeeze(preds_train_t[i]))

# import cv2
# x = cv2.imwrite("abc.jpg", np.squeeze(preds_train_t[ix]))

# # perform a sanity check on some random training samples
# ix = random.randint(0,len(preds_val_t))
# imshow(X_train[int(X_train.shape[0]*0.9):][ix])
# plt.show()
# imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
# plt.show()
# imshow(np.squeeze(preds_val_t[ix]))
# plt.show()








