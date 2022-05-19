import cv2


def GaussB(path): 
    # Gauss Binar
    gray_image = cv2.imread(path, 0)
    bina_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

    
    if __name__ == '__main__':
        cv2.imshow('BinarGauss',bina_image)
        cv2.waitKey()

    return bina_image


def GaussBED(path):
    # Gauss Binar + (erode + dilate) (noise reduction)
    gray_image = cv2.imread(path, 0)
    bina_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,2))
    bina_image = cv2.erode(~bina_image, kernel,iterations = 1)   # erode to noise reduction
    bina_image = cv2.erode(~bina_image, kernel,iterations = 1)   # dilate to restore words

    
    if __name__ == '__main__':
        cv2.imshow('BinarGauss',bina_image)
        cv2.waitKey()

    return bina_image

if __name__ == '__main__':
    GaussBED(r'Development\imageTest\einfach_table.jpg')
    # GaussBED(r'Development\imageTest\rotate_table.png')

