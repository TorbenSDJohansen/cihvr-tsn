# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:38:33 2020

@author: tsdj

The scripts contains a collection of useful functions built on either cv2
or numpy for working with images, including "cosmetic" functions such as to
draw points and show image, and more auxillary functions, such as to fit
contours or lines on images.

The entire script is mostly used directly for "identifying points" on images,
such as is used in `functional_ip.py`.

As the collection of functions are very much general purpose and the aim is to
provide excellent documentation for each, it may be a long-term goal to
build a package.

STATUS (2020-06-22):
    CURRENT FUNCTIONS: `fit_line` is very hacky (in a bad way).
    FUTURE FUNCTIONS: `recursive_draw_contours` is purely in a placeholder
    state as of yet.
"""

import cv2 # basicly "hack" for pylint to recognize member modules

import numpy as np
import random
import math

def draw_points(
        image: np.ndarray,
        points: list or tuple,
        radius: int = 10,
        color: tuple = (255, 0, 0),
    ) -> np.ndarray:
    """
    Draws points on a greyscale image and return a color image with the drawn
    points.

    Parameters
    ----------
    image : np.ndarray
        Greyscale image.
    points : list or tuple
        Iterable of points to draw.
    radius : int, optional
        Radius of drawn points. The default is 10.
    color : tuple, optional
        Color of the points. The default is (255, 0, 0), corrosponding to blue.

    Returns
    -------
    image : np.ndarray
        Color image with points.

    """
    if len(image.shape)==2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for p in points: # pylint: disable=C0103
        image = cv2.circle(image, (int(p[0]), int(p[1])), radius, color, -1)

    return image

def draw_points_BW(
        image: np.ndarray,
        points: list or tuple,
        radius: int = 10,
    ) -> np.ndarray:
    """
    Draws points on a greyscale image and return a color image with the drawn
    points.

    Parameters
    ----------
    image : np.ndarray
        Greyscale image.
    points : list or tuple
        Iterable of points to draw.
    radius : int, optional
        Radius of drawn points. The default is 10.
   
    Returns
    -------
    image : np.ndarray
        BW image with points.

    """
    #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    img= image.copy()
    for p in points: # pylint: disable=C0103
        img = cv2.circle(img, (int(p[0]), int(p[1])), radius, 255, -1)

    return img


def show(image: np.ndarray or list or tuple,w=1400,h=1000):
    """
    Draws an image or multiple images in resizeable window(s).

    Parameters
    ----------
    image : np.ndarray or list or tuple
        An image or multiple images.

    Returns
    -------
    None.

    """
    if isinstance(image, (list, tuple)):
        for i, img in enumerate(image):
            cv2.resizeWindow(f'tmp{i}', w, h)
            cv2.namedWindow(f'tmp{i}', cv2.WINDOW_NORMAL)
            cv2.imshow(f'tmp{i}', img)
    else:
        
        
        cv2.namedWindow('tmp', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('tmp', w, h)
        cv2.imshow('tmp', image)

    cv2.waitKey()
    cv2.destroyAllWindows()


def get_cell(image: np.ndarray, cell_info: dict) -> np.ndarray:
    """
    Peforms a rectangular crop of an image based on `cell_info`. Even if the
    four points specified in `cell_info` does not lead to a rectangle that is
    aligned with the bottom of the image, this is changed to the minimum size
    rectangle aligned with the bottom of the image that surrounds the cell.

    Parameters
    ----------
    image : np.ndarray
        The image to crop the cell from.
    cell_info : dict
        A dictionary of four (key, value) pairs, each with a point (tuple of
        two integers) that specify the point of a corner.

    Returns
    -------
    np.ndarray
        The cropped cell.

    """
    min_x = min(cell_info['top left'][0], cell_info['bottom left'][0])
    max_x = max(cell_info['top right'][0], cell_info['bottom right'][0])
    min_y = min(cell_info['top left'][1], cell_info['top right'][1])
    max_y = max(cell_info['bottom left'][1], cell_info['bottom right'][1])

    return image[min_y:max_y, min_x:max_x]

def show_cell(image: np.ndarray, cell_info: dict):
    """
    Peforms a rectangular crop of an image based on `cell_info`. Even if the
    four points specified in `cell_info` does not lead to a rectangle that is
    aligned with the bottom of the image, this is changed to the minimum size
    rectangle aligned with the bottom of the image that surrounds the cell.
    Further, draws the points from `cell_info` on the images. These are the
    "original" points from `cell_info`, that may not align with the bottom of
    the images.
    Finally, both the crop and the image with the drawn points are shown.

    Parameters
    ----------
    image : np.ndarray
        The image to crop the cell from and draw the points on.
    cell_info : dict
        A dictionary of four (key, value) pairs, each with a point (tuple of
        two integers) that specify the point of a corner.

    Returns
    -------
    None.

    """
    cell = get_cell(image, cell_info)
    points = list(cell_info.values())
    image_with_points = draw_points(image, points)

    show([cell, image_with_points])


def find_contours(image: np.ndarray) -> list:
    """
    Finds and returns all contours in an image.

    Parameters
    ----------
    image : np.ndarray

    Returns
    -------
    list
        All individual contours found in the image.

    """
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def dilate(image: np.ndarray, kernel: tuple, iterations: int) -> np.ndarray:
    """
    Performs dilation using a rectangle kernel with size as specified a
    specified number of times on an image.

    Parameters
    ----------
    image : np.ndarray
        The image to erode.
    kernel : tuple
        The size of the kernel, i.e. a tuple of two integers.
    iterations : int
        The number of iterations to perform.

    Returns
    -------
    image : np.ndarray
        The dilated image.

    """
    dil_structure = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
    image = cv2.dilate(image, kernel=dil_structure, iterations=iterations)

    return image


def erode(image: np.ndarray, kernel: tuple, iterations: int) -> np.ndarray:
    """
    Performs erosion using a rectangle kernel with size as specified a
    specified number of times on an image.

    Parameters
    ----------
    image : np.ndarray
        The image to erode.
    kernel : tuple
        The size of the kernel, i.e. a tuple of two integers.
    iterations : int
        The number of iterations to perform.

    Returns
    -------
    image : np.ndarray
        The eroded image.

    """
    dil_structure = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
    image = cv2.erode(image, kernel=dil_structure, iterations=iterations)

    return image


def erode_dilate_chain(image: np.ndarray, info: list) -> np.ndarray:
    """
    The function allows a chain of erosions and/or dilations to be performed
    on an image, with complete flixibility on the order, kernel size, and
    iterations of the chain. It works by going through elements in `info`,
    which controls whether to perform erosion or dilation and the kernel size
    and the number of iterations of the given step.

    The kernels applied are always rectangular.

    Below is an example on how `info` could be structured and what the function
    would do in this case.

    Let:
        info = [
            ('erode', ((3, 3), 2)),
            ('dilate', ((3, 3), 3)),
            ('erode', ((5, 5), 1)),
            ]
    In this case, this functions applies two iterations of erosion with kernel
    size (3, 3) to the image, then three iterations of dilation with kernel
    size (3, 3) to the image, and then one step of erosion to the image with
    kernel size (5, 5).

    Parameters
    ----------
    image : np.ndarray
        The image to apply the erosions and dilations.
    info : list
        Info on how to perform the steps. See the example above.

    Returns
    -------
    image : np.ndarray
        The image after the chain of erosions and/or dilations.

    """
    flow = {
        'dilate': dilate,
        'erode': erode,
        }
    for method, params in info:
        image = flow[method](image, *params)

    return image


def extract_features(image: np.ndarray, nb_points: int, min_dist: int) -> np.ndarray:
    """
    Uses `cv2.goodFeaturesToTrack` to extract corners from an image.

    Parameters
    ----------
    image : np.ndarray
        The image where the corners are located.
    nb_points : int
        The (maximum) number of corners to extract.
    min_dist : int
        The minimum distance between the corners.

    Returns
    -------
    corners : np.ndarray
        Array of the corners.

    """
    corners = cv2.goodFeaturesToTrack(
        image,
        nb_points,
        qualityLevel=0.5,
        minDistance=min_dist,
        )
    corners = np.reshape(np.int0(corners), (corners.shape[0], 2))
    corners = corners[np.argsort(corners[:, 1]), :]

    return corners


def crop(image: np.ndarray, info: dict) -> np.ndarray:
    """
    Crop out a part of an image by cropping out a share of the top, bottom,
    left, and right, as specified in `info`.

    Parameters
    ----------
    image : np.ndarray
        The image to crop from.
    info : dict
        Dictionary with four (key, value) pairs, each specifying how much to
        crop from a given side (as a SHARE, i.e. 0.1 = 10%).

    Returns
    -------
    np.ndarray
        The crop of the image.

    """
    height, width = image.shape[:2]

    min_height = int(height * info['crop_top'])
    max_height = int(height * (1 - info['crop_bottom']))

    min_width = int(width * info['crop_left'])
    max_width = int(width * (1 - info['crop_right']))

    return image[min_height:max_height, min_width:max_width]


def fit_line(contour: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    The functions fits a line with an M-estimator (a simple and fast least
    squares method) on the points of thecontour. The line is then drawn on the
    mask, which is then returned. The mask may not be entirely blank - this
    function is highly useful for drawing multiple lines on a mask that is
    then gradually filled with lines. It is used this way in `fit_contours`.

    FIXME
    The function does work fine, but its implementation is very "hacky" and
    difficult to read. It may be beneficial to change its implemenation.

    Parameters
    ----------
    contour : np.ndarray
        The contour (points) used to estimate the line.
    mask : np.ndarray
        The mask on which the line is drawn.

    Returns
    -------
    mask : TYPE
        The mask with the line drawn on it.

    """
    if isinstance(mask, cv2.UMat):
        mask = mask.get()
    width = mask.shape[1]

    vx, vy, x, y = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01) # pylint: disable=C0103
    # from official documentation: (vx, vy, x0, y0), where (vx, vy) is a
    # normalized vector collinear to the line and (x0, y0) is a point on the
    # line.

    # These go way outside the image and are then used to help define the two
    # points used to draw the line.
    # There must be a smarter way then to extend outside the image...
    lefty = int((-x * vy / vx) + y)
    righty = int(((width - x) * vy / vx) + y)

    # Below ensures near infinite slope doesn't break the program. But very
    # "hacky" implementation, so it may be beneficial to change it.
    int32max = 2147483647
    if lefty > int32max:
        lefty = int32max
    if lefty < -int32max:
        lefty = -int32max
    if righty > int32max:
        righty = int32max
    if righty < -int32max:
        righty = -int32max

    # Connects two POINTS with a line.
    mask = cv2.line(mask, (width - 1, righty), (0, lefty), 255, 20)

    return mask


def fit_contours(image: np.ndarray) -> np.ndarray:
    """
    The function finds all contours on an image and then fits lines for each
    contour which are then drawn on a black mask. The lines are fitted with an
    M-estimator (a simple and fast least squares method) on the points of the
    contour. The drawn lines are thin (width 1).

    A typical use case is to transform a grid into a thin grid. That is, a
    vertical grid can be passed as the image and a vertical grid with each line
    thin is then returned, to provide one specific example.

    Parameters
    ----------
    image : np.ndarray
        The image from which the contours are found.

    Returns
    -------
    mask : np.ndarray
        The mask with the lines drawn on it.

    """
    mask = np.zeros_like(image)
    contours = find_contours(image)
    for contour in contours:
        mask = fit_line(contour, mask) # # pylint: disable=C0103


    if isinstance(mask, cv2.UMat):
        mask = mask.get()

    return mask


def draw_contours(
        image: np.ndarray,
        erode_dilate_info: list,
        width_limits: tuple,
        corner_distances: dict,
        vert: bool,
        ) -> np.ndarray:
    """
    The function draws the contours of an image on a mask if they meet certain
    criteria with respect to their location and size.
    It works by initially applying a chain of erosions and/or dilations to the
    image (such as to remove small clutters by erosion or to link clutters
    by dilation) and then finding all contours of the eroded and/or dilated
    image. It then loops over each contour, and if they meet criteria with
    respect to their size and placement they are drawn on a mask.

    The function is primarily designed to be used to "refine" a raw vertical
    or horizontal grid into a higher quality one. This is also the reason for
    the `vert` option, the purpose of which is to transpose a vertical grid so
    it can be treated similarly to a horizontal grid.

    Parameters
    ----------
    image : np.ndarray
        The image where the contours are found from (after erosion/dilation).
    erode_dilate_info : list
        Info on how to perform the erosion and/or dilation steps. To truly
        understand its implementaiton, the reader is referred to the example
        in the documentation of `erode_dilate_chain`.
    width_limits : tuple
        A tuple of two integers, respectively the minimum and maximum allowed
        width of the contours which are drawn.
    corner_distances : dict
        A dictionary of four (key, value) pairs, each containing an integer
        specifying the distance between the contour and an edge required for
        the contour to be drawn.
    vert : bool
        Whether the image is "vertical". Vertical images are transposed. This
        is useful to treat vertical grids the same way as horizontal grid.

    Returns
    -------
    mask : np.ndarray
        A mask with each contour meeting the criteria drawn on it.

    """
    image = erode_dilate_chain(image, erode_dilate_info)

    if vert:
        image = np.transpose(image)

    mask = np.zeros_like(image)

    contours = find_contours(image)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour) # pylint: disable=C0103, W0612
        # WHY NOT USE h?
        # h is always the "thickness" of a line. The current implementation is
        # to discard "too thin" lines by erosion, and hence h really plays no
        # role in this implementation. With that said, it could at a later
        # stage become useful if either a) we no longer use erosion for
        # deletion of "too thin" lines or b) to discard too thick lines.
        if (
                width_limits[1] >= w >= width_limits[0] and
                corner_distances['bottom'] >= y >= corner_distances['top'] and
                corner_distances['right'] >= x >= corner_distances['left']
            ):
            mask = cv2.drawContours(mask, [contour], 0, 255, -1)

    if isinstance(mask, cv2.UMat):
        mask = mask.get()

    if vert:
        mask = np.transpose(mask)

    return mask


def recursive_draw_contours(
        image: np.ndarray,
        # erode_dilate_info: list,
        width_limits: tuple,
        corner_distances: dict,
        vert: bool,
        nb_lines: int,
        ) -> np.ndarray:
    # TODO placeholder function to be implemented
    # Version of `draw_contours`, but erosion (and maybe also dilation?) is
    # gradually changed until "right" number of contours are left.

    found_lines = None

    while found_lines != nb_lines:
        # do stuff to construct chain
        # call draw_contours()
        # count number of lines (contours)
        # maybe first count after thin lines drawn..? probably NOT
        # now logic. If few lines, less erosion. If too many lines, more.
        # HOWEVER, also remember that it may be impossible to solve. So check
        # if "just above" AND "just below" are triggered. Break in this case, and
        # somehow let it be known that flag is bad (use flag and OR it with a false here, otherwise a true)
        pass

    # when not possible to find correct number, can we use 2x2 kernels..?

    raise NotImplementedError
    # return mask


def _run(fname):
    #fname = 'Y:/RegionH/SPJ/Journals_jpg/1962/2014-08-18/SP2_00004.pdf.page-0.jpg'
    image = cv2.imread(fname, 0)
    from ce.sweep import grid as gl
    params = {
        'crop_info': {
            'crop_top': 0.45,
            'crop_bottom': 0.15,
            'crop_left': 0.0,
            'crop_right': 0.0,
            },
        }
    image = crop(image=image, info=params['crop_info'])
    height, width = image.shape[:2]
    vgrid, hgrid = gl.getBinaryLinePixels(image)
    vgrid[:,:50]=0
    vgrid[:,-50:]=0
    vgrid_sum = cv2.reduce(src=vgrid,dim=1,rtype=cv2.REDUCE_SUM,dtype=cv2.CV_32S)
    first_line = np.argmax(np.diff(vgrid_sum[:,0]))
    last_line  = np.argmin(np.diff(vgrid_sum[:,0]))
    cts = find_contours(hgrid)
    yfirstline = []
    ylastline = []
    for c in cts:
        x,y,w,h = cv2.boundingRect(c)
        yfirstline.append((first_line-y))
        ylastline.append((last_line-y))
    y1 = yfirstline.index(max([n for n in yfirstline if n<0]))
    y2 = ylastline.index(min([n for n in ylastline if n>0]))        
    mod_vgrid = draw_contours(
        image=vgrid,
        erode_dilate_info=[
            ('erode', ((1, 1), 1)),
            ('dilate', ((3, 3), 2)),
            # ('dilate', ((1, 100), 1)),
            ],
        width_limits=(0, 10000),
        corner_distances={
            'top': int(width * 0.02), # to remove lines in 10% most left
            'bottom': width - int(width * 0.02), # to remove lines in 10% most right
            'left': int(height * 0.01),
            'right': height - int(height * 0.01),
            },
        vert=True, # If True, transpose so it acts as a horizontal grid.
        )

    thin_vgrid = fit_contours(mod_vgrid) # pylint: disable=C0103
    thin_vgrid =    fit_line(cts[y1], thin_vgrid)   
    thin_vgrid =    fit_line(cts[y2], thin_vgrid)   
    
    corns = extract_features(thin_vgrid, 200,20) 
    img = draw_points(image,corns)
    show(img)

def add_circles(target_points: np.ndarray):

    _s = target_points[:,1].argsort()
    _s = target_points[_s[-30:],:]
    circlepoints= []
    for s in _s:    
        circle_x = s[0]
        circle_y = s[1]
        for iter in range(100):
            r, theta = [math.sqrt(random.randint(0,10))*math.sqrt(10), 2*math.pi*random.random()]
            circlepoints.append((r * math.cos(theta) + circle_x,r * math.sin(theta) + circle_y))
    circs = np.asarray(circlepoints)
    return circs


if __name__ == '__main__':
    #fname_d = 'Y:/RegionH/SPJ/Journals_jpg/1960/2014-05-23/SPJ_2014-05-23_0001.PDF.page-0.jpg'
    #fname = 'Y:/RegionH/SPJ/Journals_jpg/1960/2014-05-23/SPJ_2014-05-23_0002.PDF.page-0.jpg'
    #fname_d = 'Y:/RegionH/SPJ/Journals_jpg/1959/2014-03-31/SPJ_2014-03-31_0015.PDF.page-0.jpg'
    # fname = 'Y:/RegionH/SPJ/Journals_jpg/1960/2014-05-23/SPJ_2014-05-23_0003.PDF.page-0.jpg'
    # for i in range(19):
    #     _run(fname)
    pass






# import matplotlib.pyplot as plt

# plt.figure()
# plt.plot(range(1180),vgrid_sum[:,0])
# plt.show()
# import cv2
# vgrid_sum = cv2.reduce(src=vgrid,dim=1,rtype=cv2.REDUCE_SUM,dtype=cv2.CV_32S)
# image =vgrid
