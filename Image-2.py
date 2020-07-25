#Program to read and dispplay an img

import cv2

img=cv2.imread('dog.png')
gray=cv2.imread('dog.png',cv2.IMREAD_GRAYSCALE)
cv2.imshow("Dog Image",img)
cv2.imshow("Gray Image",gray)
cv2.waitKey(0)  #wait infinitely to destroy all windows
cv2.destroyAllWindows()
