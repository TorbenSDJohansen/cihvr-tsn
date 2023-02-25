import os
import collections
import operator
import itertools
import json
import reprlib
import xml.etree.ElementTree as ET
import pathlib

from typing import NamedTuple, Tuple
from collections.abc import MutableMapping

import numpy as np
import cv2 as cv

from probreg import filterreg

from utils import flatten_list, lists_to_tuples 
from pointcloud import PointCloud


def _nonzero_pixels(array): 
    return list(zip(*array.nonzero()))


def _fill_lines(lines, n):
    '''
    Fill template lines with pixels, as we are only given end points as
    opposed to the KeypointDetector, which has all pixels within an identified
    line
    '''

    pixels_per_line = n // len(lines)

    def _fill_line(line):
        line_x = np.linspace(line[0][0], line[1][0], pixels_per_line)
        line_y = np.linspace(line[0][1], line[1][1], pixels_per_line)
        return np.array(list(zip(line_x, line_y))).astype(int)

    return [_fill_line(line) for line in lines]


def _unpack_keypoints(lines, n_keypoints):
    filled = np.array(_fill_lines(lines, n_keypoints))
    return filled.reshape(filled.shape[0] * filled.shape[1], 2)


class Template:
    '''
    Wrapper around necessary information related to the chosen Template image.
    Includes point clouds as well as image dimensions.
    '''
    def __init__(self, width, height, lines, n_keypoints=30):
        self._width = width
        self._height = height
        self._lines = lines

        self._n_keypoints = n_keypoints

        self._filled_lines = _unpack_keypoints(lines, n_keypoints)

        self.point_cloud = PointCloud(np.array(self._filled_lines))

    def __repr__(self):
        class_name = type(self).__name__
        return '{}(width={!r}, height={!r}, points={!r})'.format(
                class_name, self._width, self._height, self.point_cloud) 

    def draw_on_image(self, image, color=(0, 0, 255)):
        return self.point_cloud.draw_on_image(image, color=color)

    @property
    def dim(self):
        return (self._width, self._height)


    @classmethod
    def from_xml(cls, xml, n_keypoints=30):
        def _unpack_object(obj):
            pts = [x.text for x in list(obj[-1])]
            xmin, ymin, xmax, ymax = pts
            return [[int(xmin), int(ymin)], [int(xmax), int(ymax)]]

        def _lines(root):
            lines = []
            for line in root.findall('object'):
                lines.append(_unpack_object(line))
            return lines

        root = ET.parse(xml).getroot()
        return cls(*_xml_size(root), _lines(root), n_keypoints)

    @property
    def lines(self):
        return self._lines


def crop_template(template, crop_info):
    pc = template.point_cloud.points
    h, w = template._height, template._width
    min_h = int(h * crop_info['top'])
    max_h = int(h * (1 - crop_info['bot']))
    min_w = int(w * crop_info['left'])
    max_w = int(w * (1 - crop_info['right']))    
    boundary = lambda x: (min_w < x[0] < max_w) and (min_h < x[1] < max_h)
    idxes = list(map(boundary, pc))
    template.point_cloud = PointCloud(pc[idxes])
    template._height = int(max_h - min_h)
    template._width = int(max_w - min_w)
    return template  

def crop_pointcloud(target,keypoints, crop_info):
    pc = keypoints.points
    h, w = target.shape[:2]
    min_h = int(h * crop_info['top'])
    max_h = int(h * (1 - crop_info['bot']))
    min_w = int(w * crop_info['left'])
    max_w = int(w * (1 - crop_info['right']))    
    boundary = lambda x: (min_w < x[0] < max_w) and (min_h < x[1] < max_h)
    idxes = list(map(boundary, pc))
    pp = PointCloud(pc[idxes])
    return pp 

class Overlay(collections.UserDict):
    # TODO: Create a transform method such that the overlay
    # knows how to transform itself. It will be called by PointDrift
    # and it will return a TransformedOverlay object that knows how
    # to draw itself as well as write cells to disk.

    # This class has to be static, as the overlay itself won't change
    # and therefore we need a separate class for transformed overlays

    def __init__(self, width, height, cells=None):
        self.data = cells or {}
        self._width = width
        self._height = height

        self.__has_setup = False

    @classmethod
    def from_xml(cls, xml):
        root = ET.parse(xml).getroot()
        parsed = [_parse_xml_object(cell) for cell in root.findall('object')]
        return cls(*_xml_size(root), {name:cell for name, cell in parsed})

    def _setup_dirs(self, root):
        if not os.path.exists(root):
            os.mkdir(root)
        for key in self.data.keys():
            subdir = os.path.join(root, key)
            if not os.path.exists(subdir):
                os.mkdir(subdir)

        self.__has_setup = True

    def draw_on_image(self, image):
        canvas = image.copy()
        for cell in self.values():
            cell.draw_on_image(canvas)
        return canvas

    def write_cells(self, image, image_name, root='./output'):
        if not self.__has_setup:
            self._setup_dirs(root)

        for name, cell in self.data.items():
            dst = os.path.join(root, name)
            checked = _check_boundaries(cell, image)
            crop = checked.crop_from_image(image)
            cv.imwrite(os.path.join(dst, image_name + '.jpg'), crop)

    def transform(self, parameters):
        transformed = {}
        for name, cell in self.items():
            transformed[name] = cell.transform(parameters)
        return TransformedOverlay(self._width, self._height, transformed)


class TransformedOverlay(collections.UserDict):

    def __init__(self, width, height, cells):
        self.data = cells
        self._width = width
        self._height = height

        self.__has_setup = False

    def draw_on_image(self, image):
        canvas = image.copy()
        for cell in self.values():
            cell.draw_on_image(canvas)
        return canvas

    def _setup_dirs(self, root):
        if not os.path.exists(root):
            os.mkdir(root)
        for key in self.data.keys():
            subdir = os.path.join(root, key)
            if not os.path.exists(subdir):
                os.mkdir(subdir)

        self.__has_setup = True

    def write_cells(self, image, image_name, root='./output'):
        if not self.__has_setup:
            self._setup_dirs(root)

        for name, cell in self.data.items():
            dst = os.path.join(root, name)
            checked = _check_boundaries(cell, image)
            crop = checked.crop_from_image(image)
            cv.imwrite(os.path.join(dst, image_name + '.jpg'), crop)


def _check_boundaries(cell, image):
    xmin = max(cell.xmin, 0)
    ymin = max(cell.ymin, 0)
    xmax = min(image.shape[1], cell.xmax)
    ymax = min(image.shape[0], cell.ymax)
    return Cell(cell.name, xmin, ymin, xmax, ymax)


def _xml_size(root):
    size = list(root.find('size'))
    width, height = size[0].text, size[1].text
    return int(width), int(height)


def _parse_xml_object(obj):
    name = obj.find('name').text
    box = obj.find('bndbox')
    return name, Cell(name, *_parse_xml_box(box))


def _parse_xml_box(box): return [int(stat.text) for stat in box]


class Cell(NamedTuple):
    name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def top_left(self):
        return np.array([self.xmin, self.ymin], dtype=int)

    @property
    def bot_right(self):
        return np.array([self.xmax, self.ymax], dtype=int)

    @property
    def size(self):
        return self.width * self.height

    @property
    def width(self):
        return int(self.xmax - self.xmin)

    @property
    def height(self):
        return int(self.ymax - self.ymin)
    
    def crop_from_image(self, image):
        return image[self.ymin:self.ymax, self.xmin:self.xmax]

    def transform(self, parameters):
        tl_transform = parameters.rotation @ self.top_left + parameters.translation
        br_transform = parameters.rotation @ self.bot_right + parameters.translation
        xmin, ymin = int(tl_transform[0]), int(tl_transform[1])
        xmax, ymax = int(br_transform[0]), int(br_transform[1])
        return Cell(self.name, xmin, ymin, xmax, ymax)

    def draw_on_image(self, image):
        cv.rectangle(image, (self.xmin, self.ymin), (self.xmax, self.ymax), (0, 0, 255), 10)
