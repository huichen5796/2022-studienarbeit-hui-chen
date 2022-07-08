# hier ist ein Beispiel für Objekt Detection
# Quelle: https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection


import cv2
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

COCO_INSTANCE_CATEGORY_NAMES = [
    '__BACKGROUND__', 'person', 'bicycle', 'car', 'motorcycle',
    'airplane', 'bus', 'train', 'trunk', 'boat', 'traffic light',
    'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A',
    'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
# https://pytorch.org/vision/stable/models.html

COCO_PERSON_KEYPOINT_NAMES = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
] # for torchvision.models.detection.keypointrcnn_resnet50_fpn(
  #                                                            pretrained=False, progress=True, 
  #                                                            num_classes=2, num_keypoints=17, 
  #                                                            pretrained_backbone=True, 
  #                                                            trainable_backbone_layers=None, **kwargs)


model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)


model.eval()
 
cap = cv2.VideoCapture(0)
transform = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)
while True:
    ret, frame = cap.read()
    image = frame
    frame = transform(frame)
    pred = model([frame])
 

    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[ii] for ii in list(pred[0]['labels'].numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
 

    pred_boxes = [[ii[0], ii[1], ii[2], ii[3]] for ii in list(pred[0]['boxes'].detach().numpy())]
 

    pred_index = [pred_score.index(x) for x in pred_score if x > 0.5]
 
    for index in pred_index:
        box = pred_boxes[index]
        cv2.rectangle(img=image, pt1=[int(box[0]), int(box[1])], pt2=[int(box[2]), int(box[3])],
                      color=(0, 0, 225), thickness=3)
        texts = pred_class[index] + ":" + str(np.round(pred_score[index], 2))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, texts, (int(box[0]), int(box[1])), font, 1, (200, 255, 155), 2, cv2.LINE_AA)
 
    plt.imshow(image)
    plt.show()

