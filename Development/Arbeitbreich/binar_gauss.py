import cv2
def BinarGauss(path):
    gray_image = cv2.imread(path, 0)
    bina_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

    # cv2.imshow('BinarGauss',bina_image)
    # cv2.waitKey()

    return bina_image

if __name__ == '__main__':
    BinarGauss(r'Development\imageTest\rotate_table.png')

