#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
    Created on Aug 10 16:50:05 2024
    @author: STRH
    Code for running SORT using YOLOv5 on image or video dataset.
'''

# import libraries
import cv2
import os
import sys
import time
import torch
import random
import argparse
import numpy as np
import os.path as osp
from pathlib import Path

# import tracker
from sort import Sort
from util import VisulizeTrack

sys.path.append('/home/setare/Vision/yolov5')

# import from yolov5
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, smart_inference_mode

# constant val
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

# selct device
device = select_device('0')
half = device.type != 'cpu'  # half precision only supported on CUDA
imgsz=640
t0, t1, t2, t3 = 0,0,0,0

# parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="SORT Inference to Evaluation")
    parser.add_argument('--weights', type=str, required=True, help="Path to detection model weights")
    parser.add_argument('--input_type', type=str, required=True, help="Input type: image or video")
    parser.add_argument('--input_path', type=str, required=True, help="Path to input images folder or video file")
    parser.add_argument('--save_mot', type=str, required=True, help="save results in mot format")
    parser.add_argument('--save_path', type=str, required=False, help="path to folder for saving results")
    parser.add_argument('--gt', type=str, required=False, help="path to gt.txt file")
    parser.add_argument('--save_video', type=str, required=True, help="if you want to save the tracking result visualization set it True")
    return parser.parse_args()


# parse args 
args = parse_args()

if args.gt:
    gt_path = args.gt
else:
    # loading model using torch.hub
    # model = torch.hub.load('ultralytics/yolov5', 'custom', path= args.weights, force_reload= False)
    # model.float()
    # model.eval()

    # load detection weights using detectmultibackend
    model = DetectMultiBackend(args.weights, device=device, dnn=False)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine

    imgsz = check_img_size(imgsz, s=stride)  # check image size
    if half:
        model.model.half()  # to FP16
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

#  get images list when input-type is image
def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def read_gt(file_path):
    # Read and parse the file
    with open(file_path, 'r') as file:
        gt_lines = file.readlines()
    
    # Initialize a dictionary to hold lists of boxes for each frame
    frames_boxes = {}
    
    # Process each line in the file
    for line in gt_lines:
        # Split the line by commas
        parts = line.strip().split(',')
        # Extract relevant fields
        frame_id = int(parts[0])
        object_id = int(parts[1])
        x = float(parts[2])
        y = float(parts[3])
        width = float(parts[4])
        height = float(parts[5])
        class_id = int(parts[7])
        visibility = float(parts[8])
        
        # Create a bounding box tuple
        bbox = [x, y, x + width, y + height, 1, class_id]
        
        # Add the bounding box to the corresponding frame's list
        if frame_id not in frames_boxes:
            frames_boxes[frame_id] = []
        frames_boxes[frame_id].append(bbox)
    
    # Let's print the first few frames to see the result
    frames_boxes_sorted = dict(sorted(frames_boxes.items()))
    return frames_boxes_sorted

def letterbox(img ,new_shape, color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)
 
def YOLOv5_detect(image,img_size,conf_thres,iou_thres):
    global t1, t3, t0, t2
    # Run inference
    img = torch.zeros((1, 3, img_size, img_size), device=device)  # init img
    t0 = time.time()
    img0 =image  # init img
    img = letterbox(img0, new_shape=(img_size,img_size))[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time.time()
    pred = model(img, augment=False, visualize=False)
    t2 = time.time()
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, False, False, max_det=1000)
    t3 = time.time()
    # Process detections
    print('{} s is YOLO detect time'.format(time.time()-t1))
    pred_lst=[];coun=0
    for i, det in enumerate(pred):  # detections per image
      if det is not None and len(det):
        coun+=1
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
        for *xyxy, conf, cls in reversed(det):
          #plot_one_box(xyxy, img0, label='Quad', color=colors[int(cls)], line_thickness=2)
          xmin=xyxy[0].cpu().detach().numpy().tolist()
          ymin=xyxy[1].cpu().detach().numpy().tolist()
          xmax=xyxy[2].cpu().detach().numpy().tolist()
          ymax=xyxy[3].cpu().detach().numpy().tolist()
          cfi=conf.cpu().detach().numpy().tolist()
          cls=int(cls.cpu().detach().numpy().tolist())
          #if cls==0:
          #  cl='Quad'
          pred_lst.append([xmin, ymin, xmax, ymax, cfi, cls])
          #cv2.imwrite('/content/'+str(coun)'.jpg',img0)
      return pred_lst
    

def main():
    # initialize the tracker
    tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)
    visualize = VisulizeTrack()

    # get images/video pathes
    file = args.input_type
    if file.lower() == "image":
        img_path = args.input_path
        if osp.isdir(img_path):
            files = get_image_list(img_path)
        files.sort()

        det_time = []
        track_time = []
        results = []
        frame_id = 1

        first_image = cv2.imread(files[frame_id-1])
        height, width, _ = first_image.shape

        if args.save_video.lower() == "true":
            Video=cv2.VideoWriter(args.save_path + f"/{args.input_path.split('/')[-2]}_draw.mp4",cv2.VideoWriter_fourcc(*'mp4v'),30,(width, height))

        for path in files:
            img = cv2.imread(path)
            t1 = time.time()

            if args.gt:
                dets = read_gt(gt_path)[frame_id]
            else:
                # using torch.hub
                # preds = model(img)

                # using another loading method
                dets = YOLOv5_detect(img, imgsz, 0.1, 0.5)

            # get dets in xyxy format
            # dets = preds.xyxy[0].cpu().numpy()  # if you are using torch.hub
            t2 = time.time()
            if len(dets) > 0:
                tracks = tracker.update(np.array(dets))

            else:   
                dets = np.empty((0, 6))  # empty N X (x, y, x, y, conf, cls)
                tracks = tracker.update(dets) # --> M X (x, y, x, y, id, conf, cls, ind)
            t3 = time.time()

            # print(dets, tracks)
            for track in tracks:
                x1, y1, x2, y2, track_id = track
                w, h = x2 - x1, y2 - y1
                conf = 1
                results.append([frame_id, track_id, x1, y1, w, h, conf, -1, -1, -1])
            
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,255), 3)
                cv2.putText(img, 
                            f'id: {int(track_id)}, conf: {conf:.2f}', 
                            (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            (0,255,255))
                
            if args.save_video.lower() == "true":
                Video.write(cv2.resize(img,(width, height)))
                    
            # break on pressing q or space
            cv2.imshow('BoxMOT detection', img)     
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') or key == ord('q'):
                break
            frame_id += 1
            det_time.append(t2-t1)
            track_time.append(t3-t2)

        if args.save_video.lower() == "true":
            Video.release()

        cv2.destroyAllWindows()

        avg_time_det = sum(det_time)/len(det_time)
        avg_time_track = sum(track_time)/len(track_time)

        fps_det = 1/avg_time_det
        fps_track = 1/avg_time_track

        print(f"Average Inference Time for Detection: {avg_time_det}, FPS: {fps_det}")
        print(f"Average Inference Time for Tracking: {avg_time_track}, FPS: {fps_track}")
    
    else:
        raise ValueError("Not Implemented, choose image!")

    if args.save_mot.lower()=="true":
            save = True
    else:
        save = False

    if save:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
            print("Save path created!")

        output_file = args.save_path + f"/{args.input_path.split('/')[-2]}.txt"
        with open(output_file, 'w') as f:
            for result in results:
                line = ",".join(map(str, result))
                f.write(line + "\n")
        print(f"Tracking results saved to {output_file}")

    
if __name__ == "__main__":
    main()