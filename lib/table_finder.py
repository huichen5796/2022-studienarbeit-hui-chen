import os
import datetime
import cv2
import numpy
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

MODEL_TABLE = {
    'densenet': 'models/densetable_90.pkl',
    'unet': 'models/unettable_85.pkl'
}

class TableFinder:
    def __init__(self, square_image, model_name, device):
        self.square_image = square_image
        self.model_name = model_name
        self.device = device
        self.square_image_size = square_image.shape[0]

    def load_model(self):
        return torch.load(MODEL_TABLE[self.model_name], map_location=torch.device(self.device))

    def transform(self, image):
        transform = A.Compose([
                    A.Resize(1024, 1024),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255),
                    ToTensorV2()
        ])

        return transform(image=image)["image"]
    
    def run_model(self, image, model):
        with torch.no_grad():
            table_pred = model(image)

            if self.model_name != 'unet':
                table_pred = torch.sigmoid(table_pred)

            return table_pred.cpu().detach().numpy().squeeze()
    
    def post_processing(self, table_pred):
        table_pred[:][table_pred[:] > 0.5] = 255.0
        table_pred[:][table_pred[:] < 0.5] = 0.0
        table_pred = table_pred.astype('uint8')

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        table_pred = cv2.erode(table_pred, kernel, iterations=1)
        table_pred = cv2.dilate(table_pred, kernel, iterations=4)
        table_pred = cv2.erode(table_pred, kernel, iterations=3)

        return table_pred
    
    def table_boundRects(self, table_pred_post):
        contours, _ = cv2.findContours(
            table_pred_post, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        table_contours = []

        for c in contours: # filter
            if cv2.contourArea(c) > 3000: 
                table_contours.append(c)

        table_boundRects = [None]*len(table_contours)
        for i, c in enumerate(table_contours):
            polyline = cv2.approxPolyDP(c, 5, True)
            table_boundRects[i] = cv2.boundingRect(polyline)

        table_boundRects = sorted(table_boundRects, key=lambda x: x[1])
        # Überlappende Tabellen werden zu einer zusammengeführt:
        mask_image = numpy.zeros((self.square_image_size, self.square_image_size, 1), numpy.uint8)

        for x, y, w, h in table_boundRects:
            size = 0
            triangle = numpy.array(
                [[x-size, y-size], [x-size, y+h+size], [x+w+size, y+h+size], [x+w+size, y-size]])
            cv2.fillConvexPoly(mask_image, triangle, 255)
        contours, _ = cv2.findContours(
            mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        table_contours = []

        for c in contours:
            if cv2.contourArea(c) > 40000:
                table_contours.append(c)

        table_boundRects = [None]*len(table_contours)
        for i, c in enumerate(table_contours):
            polyline = cv2.approxPolyDP(c, 5, True)
            table_boundRects[i] = cv2.boundingRect(polyline)

        table_boundRects = sorted(table_boundRects, key=lambda x: x[1])

        return table_boundRects
    
    def run(self):
        image = self.transform(self.square_image).to(self.device).unsqueeze(0)
        model = self.load_model()
        table_pred = self.run_model(image, model)
        table_pred_post = self.post_processing(table_pred)
        table_boundRects = self.table_boundRects(table_pred_post)

        return table_boundRects


