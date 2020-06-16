import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, Dropout, Dense, Flatten, Lambda
from keras.preprocessing.image import ImageDataGenerator
import joblib
import random
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

images = []
labels = []

with open('driving_dataset/data.txt') as txt:
	lines = txt.readlines()
	for i in range(len(lines)):
		lines[i] = lines[i].split(' ')
		images.append(lines[i][0])
		labels.append(float(lines[i][1].strip()))

TOTAL_FRAMES = len(lines)
WIDTH = 200
HEIGHT = 66
BATCH = 32

def preprocess(img):
	blurred = cv2.GaussianBlur(img, (3,3), cv2.BORDER_DEFAULT)
	resized = cv2.resize(blurred, (WIDTH, HEIGHT))
	return resized

images = np.array(images)
labels = np.array(labels)

labels = labels * (np.pi / 180)
def get_frames(ds, path, training, labels):
	x = np.empty(shape=(len(ds), HEIGHT, WIDTH, 3), dtype=np.float32)
	for i in range(len(ds)):
		if i % 1000 == 0:
			print(i)
		if training:
			rnd = random.random()
			if rnd > .25:
				x[i] = preprocess(cv2.imread(path + ds[i]))
			else:
				x[i] = preprocess(cv2.flip(cv2.imread(path + ds[i]), 1))
				labels[i] *= -1
		else:
			x[i] = cv2.resize(cv2.imread(path + ds[i]), (WIDTH, HEIGHT))
	return x

X_train, X_valid, y_train, y_valid = train_test_split(images, labels, test_size=0.3, random_state=5)

#X_train = get_frames(ds=X_train, path='driving_dataset/', training=True, labels=y_train)
#joblib.dump(X_train, 'X_train2.h5')
X_train = joblib.load('X_train2.h5')
#X_valid = get_frames(ds=X_valid, path='driving_dataset/', training=False, labels=y_valid)
#joblib.dump(X_valid, 'X_valid2.h5')
X_valid = joblib.load('X_valid2.h5')

generator = ImageDataGenerator(width_shift_range=.1,
                  height_shift_range=.1,
                  zoom_range=.2,
                  shear_range=.1)

print('training...')
def nvidia_model():
	model = Sequential()
	model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(HEIGHT, WIDTH, 3)))
	model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
	model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
	model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='elu'))
	model.add(Conv2D(64, (3, 3), activation='elu'))
	model.add(Dropout(.5))
	model.add(Flatten())
	model.add(Dense(100, activation='elu'))
	model.add(Dense(50, activation='elu'))
	model.add(Dense(10, activation='elu'))
	model.add(Dense(1))

	model.compile(loss='mse', optimizer=Adam(lr=1e-3))

	return model

model = nvidia_model()
h = model.fit_generator(generator.flow(X_train, y_train, batch_size=BATCH),
		  steps_per_epoch=len(X_train) / BATCH,
		  epochs=10,
		  validation_data=(X_valid, y_valid))

plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.xlabel('epochs')
plt.legend(['loss', 'val_loss'])
plt.show()

model.save('model3.h5')