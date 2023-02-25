from typing import NamedTuple

import numpy as np
import cv2 as cv
from probreg import filterreg, callbacks

import matplotlib.pyplot as plt


class Parameters(NamedTuple):
    rotation: np.ndarray
    translation: np.ndarray


class PointDrift:
    """
    The point drift algorithm. Sub-class and overwrite .fit to use another algorithm.

    Uses Filterreg by default. See 
    https://probreg.readthedocs.io/en/latest/probreg.html#probreg.filterreg.registration_filterreg
    """

    _init_params = {"rot": np.identity(2), "t": np.zeros(2)}

    def __init__(self, template):
        self._template = template

        self.parameters = None
        self.inv_parameters = None
        self.transformation_matrix = None
        self.inv_transformation_matrix = None

    @property
    def template_point_cloud(self):
        return self._template.point_cloud

    def _create_transformation_matrix(self, parameters):
        self.transformation_matrix = np.array([
            [parameters.rotation[0, 0], parameters.rotation[0, 1], parameters.translation[0]],
            [parameters.rotation[1, 0], parameters.rotation[1, 1], parameters.translation[1]],
            [0, 0, 1]], dtype=float)
        self.inv_transformation_matrix = np.linalg.inv(self.transformation_matrix)
        self.inv_parameters = Parameters(
                self.inv_transformation_matrix[:2, :2],
                self.inv_transformation_matrix[:2, 2]
        )

    def fit(self, target_point_cloud, iterations=30, show_fit=False):
        cbs = []
        if show_fit:
            cbs = [callbacks.Plot2DCallback(self.template_point_cloud.points,
                                         target_point_cloud.points)]
        params, _, q = filterreg.registration_filterreg(
                self.template_point_cloud.points,
                target_point_cloud.points,
                objective_type="pt2pt",
                update_sigma2=True,
                maxiter=iterations,
                callbacks=cbs,
                tol=0.001,
                tf_init_params=self._init_params
        )
        plt.show()
        self.parameters = Parameters(params.rot, params.t)
        self._create_transformation_matrix(self.parameters)
        self.q = q
        self.params = params

    def apply_transform(self, image):
        return self._warp(image)

    def _warp(self, image):
        image = np.array(image, dtype=np.uint8)
        transformed = cv.warpAffine(
                src=image,
                M=np.float32(self.inv_transformation_matrix.tolist()[:2]),
                dsize=image.shape[:2][::-1]
        )
        return transformed



