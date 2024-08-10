#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import os
import cv2
from sort import Sort
from util import VisulizeTrack
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5s6', pretrained= True)
model.float()
model.eval()

savepath = os.path.join(os.getcwd(), 'data', 'video')
vid = cv2.VideoCapture("/home/setare/Vision/Work/Tracking/Multi Object Tracking/ByteTrack/videos/palace.mp4")

sort = Sort(max_age=1, min_hits=3, iou_threshold=0.3)
visualize = VisulizeTrack()

frame_number = 0

while True:
    ret, frame = vid.read()
    preds = model(frame)
    detections = preds.pred[0].cpu().numpy()
    track_bbox_ids = sort.update(detections)

    for i in range(len(track_bbox_ids.tolist())):
        coords = track_bbox_ids.tolist()[i]
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]),  int(coords[3])
        name_idx = int(coords[4])
        text = f"ID: {str(name_idx)}"
        color = tuple(np.random.choice(range(256), size=3))
        color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, text, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imshow("Frame", frame)

    frame += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.deestroyAllWindows()

