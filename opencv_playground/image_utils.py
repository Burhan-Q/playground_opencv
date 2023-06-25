'''
OpenCV Image Utils
Author: Burhan Qaddoumi
Source: github.com/Burhan-Q
Date: 2023-05-24
'''

import numpy as np
import cv2 as cv
from pathlib import Path

###

"""
Usage
---

Parameters
---

Returns
---
"""

###

# Constants
IMAGE_FILE_TYPES = ['.png','.jpg','.jpeg','.tif','.tiff','.bmp']

# CV Constants

CV_LINETYPE = [cv.FILLED, cv.LINE_AA, cv.LINE_8, cv.LINE_4]
CV_TERMINATE_CRITERIA = [cv.TERM_CRITERIA_MAX_ITER, cv.TERM_CRITERIA_EPS, cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS]
CV_KMEANS_FLAG = [cv.KMEANS_PP_CENTERS, cv.KMEANS_USE_INITIAL_LABELS, cv.KMEANS_RANDOM_CENTERS]
CV_MORPHOLOGY_SHAPE = [cv.MORPH_RECT, cv.MORPH_CROSS, cv.MORPH_ELLIPSE]
CV_MORPHOLOGY_FUNC = [cv.MORPH_BLACKHAT, cv.MORPH_CLOSE, cv.MORPH_DILATE, cv.MORPH_ERODE, cv.MORPH_GRADIENT, cv.MORPH_HITMISS, cv.MORPH_OPEN, cv.MORPH_TOPHAT]
CV_INTERPOLATION = [
                    cv.INTER_NEAREST, # 0
                    cv.INTER_LINEAR, # 1
                    cv.INTER_CUBIC, # 3
                    cv.INTER_AREA, # 4
                    cv.INTER_LANCZOS4, # 5
                    cv.INTER_LINEAR_EXACT, # 6
                    cv.INTER_LINEAR_EXACT, # 7
                    cv.INTER_MAX, # 8
                    cv.WARP_FILL_OUTLIERS, # 8
                    cv.WARP_INVERSE_MAP # 16
                    ]

def show_im(img:np.ndarray, winname:str='image'):
    """
    Usage
    ---
    Simple function to display image in pop-up window, window destroyed after keypress

    Parameters
    ---
    img : ``numpy.ndarray``
        - Image to display, no validation performed if image is valid to show

    winname : ``str``
        - Title of window generated.
        - `default='image'`
    
    Returns
    ---
    Nothing returned, window displayed until keypress
    """
    cv.imshow(winname, img)
    cv.waitKey(0)
    cv.destroyWindow(winname)

def show_im_resize(img:np.ndarray,
                   height:int=1080,
                   scale_up:bool=False,
                   method=cv.INTER_CUBIC):
    """
    Usage
    ---

    Parameters
    ---

    Returns
    ---
    """
    orig_height, orig_width = img.shape[:2]
    scale_f = int(orig_height / height)

    if scale_up and scale_f < 1.0:
        width = int(orig_width * scale_f)
    elif scale_f >= 1.0:
        width = int(orig_width / scale_f)

    show_im(cv.resize(np.copy(img)))

    pass

def vis_contours(img:np.ndarray,
                 orig_img:np.ndarray=None,
                 heirarchy:bool=False,
                 mode=cv.RETR_EXTERNAL,
                 method=cv.CHAIN_APPROX_SIMPLE,
                 contour_idx:int=-1,
                 max_lvl:int=-1,
                 line_sz:int=1,
                 line_color:tuple=(0,255,0)):
    """
    Usage
    ---
    Use to visualize contours on provided image

    Parameters
    ---
    img : ``numpy.ndarray``
        - Input image to detect and draw contours.

    orig_img : ``numpy.ndarray``
        - Original image to visualize contours against.
        - `default=None`    
    
    heirarchy : ``bool``
        - If contours detected should also include heirarchy
        - `default=False`

    mode : ``int``
        - OpenCV contour retrival mode, see [OpenCV Contour Retrival](docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71) for additional modes
        - `default=cv.RETR_EXTERNAL`

    method : ``int``
        - OpenCV contour approximation method, see [OpenCV Contour Approxmiation](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga4303f45752694956374734a03c54d5ff) for addtional modes
        - `default=cv.CHAIN_APPROX_SIMPLE`
    
    contour_idx : ``int``
        - Index of contour to draw on image. Use `-1` to draw all contours found.
        - `default=-1` #returns all by default

    max_lvl : ``int``
        - Requires use of `heirarchy` and `mode` that returns nested contours.
        - `default=-1` #returns all by default

    line_sze : ``int``
        - Line thickness of the contours drawn on the image, use `cv.FILLED` to fill inside area of contours.
        - `default=1`

    line_color : ``tuple``
        - Color of drawn contour lines in BGR color format.
        - `default=(0,255,0)` #green

    Returns
    ---
    OpenCV image (using BGR color) with all contours detected draw onto image using settings provided. 
    """

    if img.shape[-1] != 3:
        color_img = cv.cvtColor(np.copy(img), cv.COLOR_GRAY2BGR)
    
    if orig_img is not None and orig_img.ndim != 3:
        draw_img = cv.cvtColor(np.copy(orig_img), cv.COLOR_GRAY2BGR)

    elif orig_img is not None and orig_img.ndim == 3:
        draw_img = np.copy(orig_img)

    cntrs, order = cv.findContours(img,mode,method)
    # order = order if heirarchy else -1
    _ = cv.drawContours(draw_img, cntrs, -1, line_color)

    return draw_img

def segment_kmeans(image:np.ndarray,
                   K:int=3,
                   iter8:int=10,
                   attempts:int=10,
                   eps_acc:float=1.0):
    """
    Usage
    ---
    Perform KMeans on provided 3-channel color image for (pseudo) image segmentation.

    Parameters
    ---
    image : ``numpy.ndarray``
        Input image for performing KMeans, converted to BGR color if grayscale.

    K : ``int``
        - Number of pixel clusters for KMeans and discrete pixel values in output image
        - `default=3`

    iter8 : ``int``
        - Termination criteria of KMeans, stops after `iter8` number of algorithm iterations OR when reaching epsilon accuracy value; whichever occurs first.
        - `default=10`

    attempts : ``int``
        Number of times the algorithm is executed using different initial labellings.
        - `default=10`.

    eps_acc : ``float``
        - Epsilon accuracy limit for termination criteria of KMeans, stops after `iter8` number of algorithm iterations OR when reaching epsilon accuracy value; whichever occurs first.
        - `default=1.0`.

    Returns
    ---
    Input image converted to BGR with `K` number of discrete pixel values.
    """

    if image.ndim <= 2:
        color = cv.cvtColor(np.copy(image),cv.COLOR_GRAY2BGR)
    elif image.ndim == 3:
        color = np.copy(image)
    else:
        assert False, f"Must use 3 channel color or gray scale image"
    
    Z = color.reshape(-1,3).astype(np.float32)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, iter8, 1.0)
    ret, label, center = cv.kmeans(Z, K, None, criteria, attempts, cv.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]

    return res.reshape((color.shape))

def gray_dnoise(image:np.ndarray,
                h:int=3,
                template:int=7,
                search:int=21):
    """
    Usage
    ---
    Denoise gray scale image using OpenCV Fast Nl Means Denoising.

    Parameters
    ---
    image : ``numpy.ndarray``
        - Input image to denoise, should be gray scale image, if color, will attempt to convert assuming input image uses BGR color.
    
    h : ``int``
        - Filtering strength parameter, smaller value perserves features but leaves noise, see more [OpenCV Denoising](https://docs.opencv.org/3.4/d1/d79/group__photo__denoise.html#ga76abf348c234cecd0faf3c42ef3dc715)
        - `default=3` #opencv default
    
    template : ``int``
        - Pixel size of window for computing weighted average for a pixel, should be odd, see more [OpenCV Denoising](https://docs.opencv.org/3.4/d1/d79/group__photo__denoise.html#ga76abf348c234cecd0faf3c42ef3dc715)
        - `default=7` #opencv default
    
    search : ``int``
        - Size in pixels of the template patch that is used to compute weights, should be odd, see more [OpenCV Denoising](https://docs.opencv.org/3.4/d1/d79/group__photo__denoise.html#ga76abf348c234cecd0faf3c42ef3dc715)
        - `default=21` #opencv default

    Returns
    ---
    Denoised image.
    """

    if image.ndim == 2:
        gray = np.copy(image)

    elif image.ndim >= 3:
        print("NOTICE: input image was not gray scale, will attempt converting to gray scale assuming BGR color image.")
        gray = cv.cvtColor(np.copy(image),cv.COLOR_BGR2GRAY)
    else:
        assert False, f"Check input to function, should be gray scale image."
    
    return cv.fastNlMeansDenoising(gray, h, template, search)

def nearest(array:np.ndarray, value:int|float, return_index:bool=False):
    """
    Usage
    ---
    Find arry value with nearest to the value input

    Parameters
    ---

    array : ``np.ndarray``
        Array to find nearest value from.

    value : ``int`` | ``float``
        Value to compare array against and find nearest term for.

    return_index : ``bool`` [Default = False]
        True, if the argment index should be returned with the result, otherwise only nearest array value returned

    Returns
    ---
    By default, `return_index=False` and only array value nearest to input value returned, otherwise, ``tuple`` of array value and array index returned.
    """

    near_idx = np.abs(array - value).argmin()
    return (array[near_idx], near_idx) if return_index else array[near_idx]

def images_from_path(
                    path_to_imgs:str|Path,
                    filename_root:str='',
                    extension:str=None
                    ):
    """
    Usage
    ---

    Parameters
    ---
    path_to_imgs : ``str`` | ``pathlib.Path``
        Directory to search for images
    
    filename_root : ``str`` [Default = '']
        Common and continuous filename root string to search for using glob-like pattern; example: '*filename*' --> filename.png, filename-1137.jpeg, _0a_filename_1j_.tif

    extension : ``str`` [Default = `None`]
        File extension for specified image types, including '.', examples: '.png', '.jpeg', '.tiff'; when not provided, searches using all image types from `image_utils.pyIMAGE_FILE_TYPES`

    Returns
    ---
    If one or more images found, returns list of image files as ``pathlib.Path`` objects based on input combination provided.
    """
    A = B = C = D = False # initialize conditions

    # Check if given path to file
    input_file = False
    input_file = Path(path_to_imgs).is_file()

    # Check directory if not file
    if not input_file:
        path_to_imgs = path_to_imgs if isinstance(path_to_imgs, Path) else Path(path_to_imgs)
    
        is_abs_path = False
        is_abs_path = path_to_imgs.is_absolute()

        assert path_to_imgs.absolute().exists(), f"""Provided path directory to images: {str(path_to_imgs)} was not found, {'check absolute path entered is correct.' if is_abs_path else 'try providing absolute path instead.'}"""

        # Conditions 
        A = filename_root == '' and extension is None
        B = filename_root == '' and extension is not None
        C = filename_root != '' and extension is None
        D = filename_root != '' and extension is not None

        # Make sure extension formatting is correct
        extension = '.' + extension if (B or D) and not extension.startswith('.') else extension

        # Get image files
        if A: # no filename root, search all extensions
            img_files = [f for f in path_to_imgs.rglob("*") if f.suffix in IMAGE_FILE_TYPES]

        elif B: # no filename root, given specific extension
            img_files = [f for f in path_to_imgs.rglob(f"*{extension}")]

        elif C: # filename root given, search all extensions
            img_files = [f for f in path_to_imgs.rglob(f"*{filename_root}*") if f.suffix in IMAGE_FILE_TYPES]

        elif D: # filename root given, given specific extension
            img_files = [f for f in path_to_imgs.rglob(f"*{filename_root}*{extension}")]

        else:
            pass

        # Check images were found
        imgs_found = len(img_files)
        assert imgs_found >= 1, f"No images found, check path provided{', filename_root, ' if filename_root != '' else ' '}and image file extensions are in {IMAGE_FILE_TYPES}"
    
    else:
        img_files = [Path(path_to_imgs)]

    return img_files
