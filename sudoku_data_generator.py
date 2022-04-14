import os
import cv2
import random

PATH = 'data_sudoku/fonts'
images = []
for file in os.listdir(PATH + '/Sample010'):
    img = cv2.imread(PATH + '/Sample010/' + file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    images.append(img)

test = random.sample(range(1016), 200)
for i in range(1016):
    if i+1 in test:
        cv2.imwrite(PATH + '/testing/9/' + str(i+1) + '.png', images[i])
    else:
        cv2.imwrite(PATH + '/training/9/' + str(i+1) + '.png', images[i])
