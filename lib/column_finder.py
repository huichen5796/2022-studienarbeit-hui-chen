import os
import datetime
import cv2
import numpy
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .image_untils import ImageUntils

MODEL_TABLE = {
    'densenet': 'models/densecol_140ss.pkl',
    'unet': 'models/unetcol_90.pkl'
}

class ColumnFinder:
    def __init__(self, square_image, model_name, device, table_boundRects):
        self.square_image = square_image
        self.model_name = model_name
        self.device = device
        self.square_image_size = square_image.shape[0]
        self.table_boundRects = table_boundRects

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
            column_pred = model(image)

            if self.model_name != 'unet':
                column_pred = torch.sigmoid(column_pred)

            return column_pred.cpu().detach().numpy().squeeze()
        
    def post_processing(self, column_pred):
        column_pred[:][column_pred[:] > 0.5] = 255.0
        column_pred[:][column_pred[:] < 0.5] = 0.0
        column_pred = column_pred.astype('uint8')

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 19))
        column_pred = cv2.erode(column_pred, kernel, iterations=2)
        column_pred = cv2.dilate(column_pred, kernel, iterations=1)

        return column_pred
    
    def cloumn_boundRects(self, column_pred):
        task_object = ImageUntils(column_pred, self.table_boundRects)
        cutted_zones = task_object.image_cut(border_color=[0])

        column_boundRects_relativ = []
        for cutted_zone in cutted_zones:
            contours, _ = cv2.findContours(
                cutted_zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            col_contours = []
            for c in contours:
                if cv2.contourArea(c) > 500:
                    x, y, w, h = cv2.boundingRect(c)
                    col_contours.append((int(x+w//2), int(w)))

            col_contours = sorted(col_contours, key=lambda x: x[0])

            n = 0
            while n <= 4:
                for i in range(len(col_contours)-1):
                    if abs(col_contours[i+1][0]-col_contours[i][0]) < 0.7*(col_contours[i+1][1]+col_contours[i][1])//2:
                        col_contours[i] = ((col_contours[i+1][0]+col_contours[i][0]) //
                                        2, (col_contours[i+1][1]+col_contours[i][1])//2)
                        col_contours[i+1] = col_contours[i]
                    else:
                        continue
                n += 1
                col_contours = list(set(col_contours))
                col_contours = sorted(col_contours, key=lambda x: x[0])
            
            for i, (col, w) in enumerate(col_contours):
                col_contours[i] = (int(col), int(w))
            
            column_boundRects_relativ.append(col_contours)
        
        return column_boundRects_relativ
    
    def run(self):
        image = self.transform(self.square_image).to(self.device).unsqueeze(0)
        model = self.load_model()
        column_pred = self.run_model(image, model)
        column_pred_post = self.post_processing(column_pred)
        column_boundRects_relativ = self.cloumn_boundRects(column_pred_post)

        return column_boundRects_relativ