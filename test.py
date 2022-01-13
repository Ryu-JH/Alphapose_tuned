"""Script for single-gpu/multi-gpu demo."""
import cv2

class drawer():
    def __init__(self, img):
        self.img = img
    
    def show(self, img):
        cv2.imshow('Demo', img)
        cv2.waitKey(30)