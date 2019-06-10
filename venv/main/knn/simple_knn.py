from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np

import argparse
import imutils
import cv2
import os
import re

base_dir = '/cats_and_dogs_small'

train_dir = os.path.join(base_dir, 'train')
train_cat_dir = os.path.join(train_dir, 'cats')
train_dog_dir = os.path.join(train_dir, 'dogs')
valid_dir = os.path.join(base_dir, 'validation')
valid_cat_dir = os.path.join(valid_dir, 'cats')
valid_dog_dir = os.path.join(valid_dir, 'dogs')
test_dir = os.path.join(base_dir, 'test')
test_cat_dir = os.path.join(test_dir, 'cats')
test_dog_dir = os.path.join(test_dir, 'dogs')

def image_to_vector(image, size=(32,32)):
    return cv2.resize(image, size).flatten()

def extract_color(image, bins = (8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])

    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    else:
        cv2.normalize(hist, hist)


    return hist.flatten()



imagePaths = list(paths.list_images(base_dir))

rawImages = []
features = []
labels = []
# lab = "/cats_and_dogs_small\\test\cats\cat.1500.jpg"
# print(lab.split("\\")[3])
for (i, imagePath) in enumerate(imagePaths):
    if "test_train" in imagePath:
        continue
    animal = imagePath.split("\\")[2]
    # print(animal)
    image = cv2.imread(imagePath)
    pixels = image_to_vector(image)
    hist = extract_color(image)

    rawImages.append(pixels)
    features.append(hist)
    labels.append(animal)


(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size=0.25, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	features, labels, test_size=0.25, random_state=42)



model = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
model.fit(trainRI, trainRL)
# model.fit(trainFeat, trainLabels)
acc = model.score(testRI, testRL)
# acc = model.score(testFeat, testLabels)

print(acc)