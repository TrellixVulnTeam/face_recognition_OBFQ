import cv2
import numpy as np
import face_recognition
imgElon=face_recognition.load_image_file('/Users/param/Desktop/elon.png')
imgTest=face_recognition.load_image_file('/Users/param/Desktop/elontest.png')
imgElon=cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest=cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
cv2.imshow('elon',imgElon)
cv2.imshow('elon',imgTest)
cv2.waitKey(0)
