from typing import NamedTuple

import numpy as np
import cv2 as cv


class PointCloud(NamedTuple):
    points: np.ndarray

    def __eq__(self, other): 
        return np.allclose(self, other, atol=5)

    def sort(self): 
        return PointCloud(self.points[self.points[:, 0].argsort()])

    def draw_on_image(self, image, size=5, color=(255, 0, 0)):
        canvas = image.copy()
        if len(canvas.shape) > 2 and canvas.shape[2] == 1:
                canvas = cv.cvtColor(canvas, cv.COLOR_GRAY2RGB)
        for p in self.points:
            cv.circle(canvas, (int(p[0]), int(p[1])), size, color, -1)
        return canvas
