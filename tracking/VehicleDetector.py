import cv2
import numpy as np
from tracking.ROIManager import ROIManager
import config

class VehicleDetector:
    def __init__(self):
        self.min_contour_area = config.MIN_CONTOUR_AREA  # Olması gereken min kontur alanını çeker

    def detect_contour(self, cleaned_mask):
        # RoiManager dan gelen kilitlenmiş roi de kontur tespit eder
        contours, hier = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        boxes = []                                      # Tespit edilen konturların koordinatlarını ekleyeceğimiz liste
        for contour in contours:
            area = cv2.contourArea(contour)             # Kontur alanını hesapalr
            if area > self.min_contour_area:            # Alan sınır değerinden büyükse
                x, y, w, h = cv2.boundingRect(contour)  # Konturun koordinatlarını tespit eder
                boxes.append((x,y,x+w,y+h))             # Konturun koordinatları listeye ekler

        return boxes                                    # Listeyi döndürür
