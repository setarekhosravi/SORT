#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import seaborn as sns
import ffmpeg
import numpy as np
import os
import cv2

def draw_border(img, pt1, pt2, color = (3,186,252), thickness = 2, r = 10, d = 20):
    x1,y1 = pt1
    x2,y2 = pt2
    whole_r_d_x = 0.3*(x2-x1)
    rx = int(0.3*whole_r_d_x)
    dx = int(whole_r_d_x-rx)
    whole_r_d_y = 0.3*(y2-y1)
    ry = int(0.3*whole_r_d_y)
    dy = int(whole_r_d_y-ry)
    ry = rx
    if dy < ry:
        dy = int(dy*rx/dx)
    # Top left
    cv2.line(img, (x1 + rx, y1), (x1 + rx + dx, y1), color, thickness)
    cv2.line(img, (x1, y1 + ry), (x1, y1 + ry + dy), color, thickness)
    cv2.ellipse(img, (x1 + rx, y1 + ry), (rx, ry), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - rx, y1), (x2 - rx - dx, y1), color, thickness)
    cv2.line(img, (x2, y1 + ry), (x2, y1 + ry + dy), color, thickness)
    cv2.ellipse(img, (x2 - rx, y1 + ry), (rx, ry), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + rx, y2), (x1 + rx + dx, y2), color, thickness)
    cv2.line(img, (x1, y2 - rx), (x1, y2 - ry - dy), color, thickness)
    cv2.ellipse(img, (x1 + rx, y2 - ry), (rx, ry), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - rx, y2), (x2 - rx - dx, y2), color, thickness)
    cv2.line(img, (x2, y2 - ry), (x2, y2 - ry - dy), color, thickness)
    cv2.ellipse(img, (x2 - rx, y2 - ry), (rx, ry), 0, 0, 90, color, thickness)
    return img

class VisulizeTrack:
    def __init__(self, unique_colors=400):
        """
        unique_colors (int): The number of unique colors (the number of unique colors dos not need to be greater than the max id)
        """
        self._unique_colors = unique_colors
        self._id_dict = {}
        self.p = np.zeros(unique_colors)
        self._colors = (np.array(sns.color_palette("hls", unique_colors))*255).astype(np.uint8)

    def _get_color(self, i):
        return tuple(self._colors[i])

    def _color(self, i):
        if i not in self._id_dict:
            inp = (self.p.max() - self.p ) + 1 
            if any(self.p == 0):
                nzidx = np.where(self.p != 0)[0]
                inp[nzidx] = 0
            soft_inp = inp / inp.sum()

            ic = np.random.choice(np.arange(self._unique_colors, dtype=int), p=soft_inp)
            self._id_dict[i] = ic

            self.p[ic] += 1

        ic = self._id_dict[i]
        return self._get_color(ic)

    def draw_bounding_boxes(self, im, bboxes: np.ndarray, ids: np.ndarray,
                        scores: np.ndarray):
        """
        im (PIL.Image): The image 
        bboxes (np.ndarray): The bounding boxes. [[x1,y1,x2,y2],...]
        ids (np.ndarray): The id's for the bounding boxes
        scores (np.ndarray): The scores's for the bounding boxes
        """

        for bbox, id_, score in zip(bboxes, ids, scores):
            color = self._color(id_)
            text = f'{id_}: {int(100 * score)}%'

            cv2.rectangle(im, (bbox[0],bbox[1]), (bbox[0]+bbox[[2]], bbox[1]+bbox[3]), color, 2)
            cv2.putText(im, text, (bbox[0],bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        return im
