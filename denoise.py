import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('trem/l/01.jpg')
#gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#height, width, _ = img.shape
dst = cv2.imread('trem/l/01.jpg')
#cv2.fastNlMeansDenoising(img,dst,15,7,21)
dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(dst)
plt.show()

