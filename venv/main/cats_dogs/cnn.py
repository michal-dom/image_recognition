from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os, shutil
from os import listdir

base_dir = '/cats_and_dogs_small'

train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

train_datagen = ImageDataGenerator(
 rescale=1./255,
 rotation_range=40,
 width_shift_range=0.2,
 height_shift_range=0.2,
 shear_range=0.2,
 zoom_range=0.2,
 horizontal_flip=True)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=32, class_mode='binary')
valid_generator = valid_datagen.flow_from_directory(valid_dir, target_size=(150, 150), batch_size=32, class_mode='binary')

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy',
 # optimizer=optimizers.RMSprop(lr=1e-4),
 # metrics=['acc'])

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
#
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100, validation_data= valid_generator, validation_steps=50)
model.save('cats_and_dogs_small_4.h5')

#
try:
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'bo', label='Dokladnosc trenowania')
    plt.plot(epochs, val_acc, 'b', label='Dokladnosc walidacji')
    plt.title('Dokladnosc trenowania i walidacji')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Strata trenowania')
    plt.plot(epochs, val_loss, 'b', label='Strata walidacji')
    plt.title('Strata trenowania i walidacji')
    plt.legend()
    plt.show()
except:
    print("oo")