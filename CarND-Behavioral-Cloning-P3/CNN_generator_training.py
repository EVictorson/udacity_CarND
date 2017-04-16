import csv
import cv2
import numpy as np
import keras
import sys
import sklearn
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Conv2D, MaxPooling2D, Cropping2D,ZeroPadding2D
from keras.layers.convolutional import Convolution2D
from PIL import Image
from sklearn.utils import shuffle
from keras.optimizers import Adam
from keras.regularizers import l2, activity_l2
from keras.layers.advanced_activations import ELU


print("==== program started ====")

def generator(samples, batch_size=32):

	# path to default training data they gave us
	path = '/home/eric/vcs/udacity/self_driving/CarND-Behavioral-Cloning-P3/linux_sim/data2/IMG/'
	num_samples = len(samples)
	#correction = 0.2 # tuneable parameter
	correction = 0.5
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):

			batch_samples = samples[offset:offset+batch_size]
			images = []
			angles = []
			
			for batch_sample in batch_samples:

				#name = path + batch_sample[0].split('/')[-1]
				name = batch_sample[1]
				#name_left =  path + batch_sample[1].split('/')[-1]
				name_left = batch_sample[2]
				#name_right = path + batch_sample[2].split('/')[-1]
				name_right = batch_sample[3]
				#center_angle = float(batch_sample[3])
				if(name == '0' or name_left == '0' or name_right == '0'):
					continue
				if(name == 0 or name_left == 0 or name_right == 0):
					continue

				center_angle = float(batch_sample[4])
				#left_angle = center_angle + correction   # tried * 0.5
				#right_angle = center_angle - correction #tried * 1.5		

				#img_center = process_image(np.asarray(Image.open(name)))
				img_center = np.asarray(Image.open(name))

				#img_left = process_image(np.asarray(Image.open(name_left)))
				#img_right = process_image(np.asarray(Image.open(name_right)))				
				#img_center_flipped = np.fliplr(img_center)
				#img_left_flipped = np.fliplr(img_left)
				#img_right_flipped = np.fliplr(img_right)

				#center_angle_flipped = -1*center_angle
				#left_angle_flipped = -1*left_angle
				#right_angle_flipped = -1*right_angle

				#images.extend((img_center, img_center_flipped, img_left, img_left_flipped, img_right, img_right_flipped))
				#angles.extend((center_angle, center_angle_flipped, left_angle, left_angle_flipped, right_angle, right_angle_flipped))
				images.append(img_center)
				angles.append(center_angle)


			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)

def process_image(image):

	zeros = np.zeros((160,320,3))
	norm_image = cv2.normalize(image, zeros, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	return norm_image

def VGG_16(weights_path=None):
    shape = (160, 320, 3)    

    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=shape))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Dropout(0.5))
    model.add(Dense(500))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    if weights_path:
        model.load_weights(weights_path)

    return model


def nvidia():

	shape = (160,320,3)
	model = Sequential()
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = shape))
	model.add(Cropping2D(cropping = ((50,20),(0,0))))
	model.add(Convolution2D(24,5,5, subsample=(2,2), activation = 'relu'))
	model.add(Convolution2D(36,5,5, subsample=(2,2), activation = 'relu'))
	model.add(Convolution2D(48,5,5, subsample=(2,2), activation = 'relu'))
	model.add(Convolution2D(64,3,3, activation = 'relu'))
	model.add(Convolution2D(64,3,3, activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dropout(0.5))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))

	return model



def lenet():
	shape = (160,320,3)
	model = Sequential()
	# crop top and bottom off image
	model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=shape))
	# image normalization and mean centering
	model.add(Lambda(lambda x: (x / 255.0) - 0.5))
	model.add(Convolution2D(6,5,5, activation='relu'))
	model.add(MaxPooling2D())
	model.add(Convolution2D(6,5,5, activation='relu'))
	model.add(MaxPooling2D())
	model.add(Convolution2D(6,5,5, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(120))
	model.add(Dropout(0.5))
	model.add(Dense(84))
	model.add(Dense(1))
	# because we are performing regression and not classification our output class is 1 with
	# no activation function

	return model


lines = []

print("==== Loading Data... ====")

#filename = 'linux_sim/data2/driving_log.csv'
filename = 'keepers_expanded.csv'


with open(filename) as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

car_images = []
steering_angles = []
length = len(lines)
print(" ==== file opened and read ====")

# let's play with adding all 3 camera images and modifying the steering angle
i = 0


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

print("==== Done Loading Data ==== ")
print("Number of Images: {}".format(len(car_images)))
print("Number of Steering Angles: {}".format(len(steering_angles)))

# select which model we will be using
print("Assigning the model")
model = nvidia()

#model = nvidia_improved()

print("Training Started...")

#model.compile(loss='mse', optimizer='adam')
model.compile(optimizer=Adam(lr=1e-4), loss='mse')

# no idea why thiswould be 13055 and not 13024 like len(train_samples) indicates
#samples_per_epoch= len(train_samples)
samples_per_epoch = 13055

history_object = model.fit_generator(train_generator, samples_per_epoch , validation_data = validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10)

model.save('model007.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

