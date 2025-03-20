import math
import numpy as np
import pandas as pd
import cv2

import torch
from api import Facedetecor as RetinaFaceDetector


def imshow_for_test(windowname, img, face_boundarys=None, landmarks=None):
    if face_boundarys is not None:
        for face_boundary in face_boundarys:
            cv2.rectangle(img, (face_boundary[0], face_boundary[1]),
                          (face_boundary[2], face_boundary[3]), (0, 0, 255), 1)
    if landmarks is not None:
        for point in landmarks:
            cv2.circle(img, (point[0].item(), point[1].item()),
                       1, (0, 0, 255), 4)
    cv2.putText(img, windowname, (15, 15),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255))
    cv2.imshow(windowname, img)
    cv2.waitKey(0)


class FaceDetector:
    def __init__(self, model_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.det = RetinaFaceDetector(model_path, device)

    def cal(self, img):
        left, top, right, bottom = self.det.get_face_box(img)
        return left, top, right, bottom


