import cv2


def NoiseReducter(img):
    # Gauss Binar + (erode + dilate) (noise reduction)
    
    bina_image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

    
    gray_image = cv2.GaussianBlur(bina_image, (3,3),0)
    
    #ret, bina_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,2))
    #bina_image = cv2.erode(~bina_image, kernel,iterations = 1)   # erode to noise reduction
    #bina_image = cv2.erode(~bina_image, kernel,iterations = 1)   # dilate to restore words

    
    if __name__ == '__main__':
        cv2.imshow('BinarGauss',bina_image)
        cv2.waitKey()

    return bina_image

if __name__ == '__main__':
    img = cv2.imread(r'Development\imageTest\textandtable_0.png', 0)
    NoiseReducter(img)

