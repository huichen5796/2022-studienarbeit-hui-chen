import cv2
import numpy as np
import matplotlib.pyplot as plt
from TableExtract import ImageRotate

img1 = cv2.imread(r'Development_tradionell\imageTest\winkel_-30.png',0)
img = cv2.bitwise_not(img1)
coords = np.column_stack(np.where(img > 0))
angle = -cv2.minAreaRect(coords)[2]   # this function has three return
                                       # [0] -- center point of recttangle
                                       # [1] -- (width, high) of rectangle
                                       # [2] -- The rotation angle of the rectangle, 
                                       # from the x-axis counterclockwise to the (high) angle, the range is (-90,0]   

# the `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we
# need to add 90 degrees to the angle
if angle < -45:
	angle = (90 + angle)
# otherwise, just take the inverse of the angle to make
# it positive
else:
	angle = angle
print(angle)

img = ImageRotate(img1, angle)

plt.imshow(img, cmap= 'gray')
plt.show()