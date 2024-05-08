from keras.src.datasets import mnist
from imutils import build_montages
import numpy as np
import cv2

from pyimagesearch.utils import make_pairs


print("[INFO] loading MNIST dataset...")
(trainX, trainY), (testX, testY) = mnist.load_data()

print("[INFO] preparing positive and negative pairs...")
(pairTrain, labelTrain) = make_pairs(trainX, trainY)
(pairTest, labelTest) = make_pairs(testX, testY)

images = []

for i in np.random.choice(np.arange(0, len(pairTrain)), size=(49,)):
    imageA, imageB = pairTrain[i]
    label = labelTrain[i]

    output = np.zeros((36, 60), dtype="uint8")
    pair = np.hstack([imageA, imageB])
    output[4:32, 0:56] = pair

    text, color = ("neg", (0, 0, 255)) if label[0] == 0 else ("pos", (0, 255, 0))

    vis = cv2.merge([output] * 3)
    vis = cv2.resize(vis, (96, 51), interpolation=cv2.INTER_LINEAR)
    cv2.putText(vis, text, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    images.append(vis)

montage = build_montages(images, (96, 51), (7, 7))[0]

cv2.imshow("Siamese Image Pairs", montage)
cv2.waitKey(0)
