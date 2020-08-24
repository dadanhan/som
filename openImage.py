import cv2
import numpy as np

im = cv2.imread('test1.tif',-1)
print(im)

cv2.imshow('test',im)
cv2.waitKey(0)