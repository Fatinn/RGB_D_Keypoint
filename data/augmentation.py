
import numpy as np
import random
import math
import cv2

class DataAugmentation:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, rgb, depth, points):
        if random.random() < self.prob:
            rgb = np.fliplr(rgb).copy()
            depth = np.fliplr(depth).copy()
            h, w = rgb.shape[:2]
            for i in range(len(points)):
                points[i] = (w - 1 - points[i][0], points[i][1])

        if random.random() < self.prob:
            rgb = np.flipud(rgb).copy()
            depth = np.flipud(depth).copy()
            h, w = rgb.shape[:2]
            for i in range(len(points)):
                points[i] = (points[i][0], h - 1 - points[i][1])

        if random.random() < self.prob:
            angle = random.uniform(-15, 15)
            h, w = rgb.shape[:2]
            center = (w / 2, h / 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            rgb = cv2.warpAffine(rgb, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
            depth = cv2.warpAffine(depth, rot_mat, (w, h), flags=cv2.INTER_LINEAR)

            cos_angle = math.cos(math.radians(angle))
            sin_angle = math.sin(math.radians(angle))
            for i in range(len(points)):
                x, y = points[i]
                x -= center[0]
                y -= center[1]
                new_x = x * cos_angle - y * sin_angle
                new_y = x * sin_angle + y * cos_angle
                points[i] = (new_x + center[0], new_y + center[1])

        if random.random() < self.prob:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            rgb = cv2.convertScaleAbs(rgb, alpha=contrast, beta=brightness)

        return rgb, depth, points
