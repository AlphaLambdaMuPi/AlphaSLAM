import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('../data/testimg/b001.jpg')
img2 = cv2.imread('../data/testimg/b005.jpg')

img1 = cv2.resize(img1, (480, 640), interpolation=cv2.INTER_CUBIC)
img2 = cv2.resize(img2, (480, 640), interpolation=cv2.INTER_CUBIC)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

plt.figure(1)
plt.imshow(abs(gray1-gray2), cmap='gray')
# plt.imshow(gray1, cmap='gray')
# plt.imshow(img1[:,:,::-1], cmap = 'gray', interpolation = 'bicubic')
# plt.scatter(p1[:, 0, 0], p1[:, 0, 1], s=err*2)

# plt.figure(2)
# plt.imshow(gray2, cmap='gray')
# plt.imshow(img2[:,:,::-1], cmap = 'gray', interpolation = 'bicubic')
# plt.scatter(p2[:, 0, 0], p2[:, 0, 1], s=err*2)

plt.show()
