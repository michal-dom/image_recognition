
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models

from keras import optimizers
from sklearn.metrics import classification_report, confusion_matrix

import os

test_folder = "/cars_dataset_greyscale/test/"

img_rows = 150
img_cols = 150
epochs = 30
batch_size = 32
num_of_train_samples = 3000
num_of_test_samples = 600


test_datagen = ImageDataGenerator(rescale=1./255)
#
test_generator = test_datagen.flow_from_directory(
    test_folder,
    class_mode="categorical",
    color_mode="grayscale",
    target_size=(150, 150),
    batch_size=batch_size
)

fnames = test_generator.filenames
nb_sample = len(fnames)

# print(fnames)


# label_test = os.listdir("/cars_dataset/test/golf/")[:10]
# label_test.extend(os.listdir("/cars_dataset/test/bmw_seria3/")[:10])
# label_test.extend(os.listdir("/cars_dataset/test/audi_a4/")[:10])

# print(len(label_test))
# print(label_test)


model = models.load_model("cars_cnn_d.h5")
model.compile(loss='categorical_crossentropy',
 optimizer=optimizers.RMSprop(lr=1e-4),
 metrics=['categorical_accuracy'])

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('dokładność podczas testowania:', test_acc)