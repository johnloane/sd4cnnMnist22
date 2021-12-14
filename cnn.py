import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import random
import requests
from PIL import Image
import cv2
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.models import Model


def letNet_model():
    model = Sequential()
    # Output from this convolution layer is 30 24x24 feature matrices
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    # Output will be  30 12x12 matrices
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Output will be 15 10x10 matrices
    model.add(Conv2D(15, (3, 3), activation='relu'))
    #Output will be 15 5x5
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(learning_rate = 0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


np.random.seed(0)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
assert(X_train.shape[0] == y_train.shape[0]), "The number of images and labels does not match for the training set"
assert(X_test.shape[0] == y_test.shape[0]), "The number of images and labels does not match for the test set"
assert(X_train.shape[1:] == (28, 28))
assert(X_test.shape[1:] == (28, 28))

num_classes = 10

# One hot encode training and test labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Normalise all values to get values between 0 and 1
X_train = X_train/255
X_test = X_test/255

print(X_train.shape)

# For Convolution we want to leave the data as a matrix

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

model = letNet_model()
print(model.summary())

h = model.fit(X_train, y_train, epochs=10, validation_split=0.1, batch_size=400, verbose=1, shuffle=1)
plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.legend(['accuracy', ['val_accuracy']])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

url = 'https://colah.github.io/posts/2014-10-Visualizing-MNIST/img/mnist_pca/MNIST-p1815-4.png'
response = requests.get(url, stream=True)
img = Image.open(response.raw)
img.show()

img_array = np.asarray(img)
print(img_array.shape)
resized = cv2.resize(img_array, (28, 28))
gray_scale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
print(gray_scale.shape)

plt.imshow(gray_scale, cmap=plt.get_cmap("gray"))
plt.show()

image = cv2.bitwise_not(gray_scale)
image = image/255
print(image.shape)
image = image.reshape(1, 28, 28, 1)
prediction = np.argmax(model.predict(image), axis=1)
print("Predicted digit: ", str(prediction))

score = model.evaluate(X_test, y_test, verbose=1)
print("Test score: ", score[0])
print("Test accuracy: ", score[1])

layer1 = Model(inputs=model.layers[0].input, outputs=model.layers[0].output)
layer2 = Model(inputs=model.layers[0].input, outputs=model.layers[2].output)

visual_layer1 = layer1.predict(image)
visual_layer2 = layer2.predict(image)

print(visual_layer1.shape)
print(visual_layer2.shape)

plt.figure(figsize=(10, 6))
for i in range(30):
    plt.subplot(6, 5, i+1)
    plt.imshow(visual_layer1[0, :, :, i], cmap=plt.get_cmap('jet'))
    plt.axis('off')
plt.show()

plt.figure(figsize=(10, 6))
for i in range(15):
    plt.subplot(3, 5, i+1)
    plt.imshow(visual_layer2[0, :, :, i], cmap=plt.get_cmap('jet'))
    plt.axis('off')
plt.show()













