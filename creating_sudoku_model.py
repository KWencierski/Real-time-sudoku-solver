from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

BATCH_SIZE = 128

train_data_dir = 'data_sudoku/fonts/training'
test_data_dir = 'data_sudoku/fonts/testing'

train_data_gen = ImageDataGenerator(rescale=1/255)
test_data_gen = ImageDataGenerator(rescale=1/255)

train_generator = train_data_gen.flow_from_directory(train_data_dir, color_mode='grayscale', target_size=(28, 28),
                                                     batch_size=BATCH_SIZE, class_mode='categorical')
test_generator = test_data_gen.flow_from_directory(test_data_dir, color_mode='grayscale', target_size=(28, 28),
                                                   batch_size=BATCH_SIZE, class_mode='categorical')

# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#
# train_images = train_images.reshape((60000, 28, 28, 1))
# train_images = train_images.astype('float32') / 255
#
# test_images = test_images.reshape((10000, 28, 28, 1))
# test_images = test_images.astype('float32') / 255

model = models.Sequential()
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(9, activation='softmax'))
model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)

# model.fit(train_images, train_labels, epochs=10, batch_size=64)
model.fit_generator(train_generator, steps_per_epoch=np.floor(train_generator.n/BATCH_SIZE), epochs=10,
                    validation_data=test_generator, validation_steps=np.floor(test_generator.n/BATCH_SIZE))

# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print('test_acc:', test_acc)

if os.path.isfile('models/model.h5') is False:
    model.save('models/model2.h5')
