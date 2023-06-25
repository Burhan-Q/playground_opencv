'''
OpenCV Calibration Utils
Author: Burhan Qaddoumi
Source: github.com/Burhan-Q
Date: 2023-06-08
'''

from pathlib import Path
import numpy as np
import cv2 as cv
from image_utils import show_im, images_from_path

def visualize_corners(color_img:np.ndarray,size:tuple|list|np.ndarray,found:bool,points:np.ndarray,scale=1.0):
    """
    Usage
    ---
    Generates visualization of checkerboard points on copy on input image, also scales point-coordinates based on scale-factor.

    Parameters
    ---
    color_img : ``np.ndarray``
        Input image that should be 3-channel RGB color image.

    size : ``tuple`` | ``list`` | ``np.ndarray``
        Count of inner-corners for checkerboard pattern `[points-per-row, points-per-column]`.

    found : ``bool``
        Pass in variable denoting if corners were detected.

    points : ``np.ndarray``
        Detected corner points if `found = True`
    
    scale : ``int`` | ``float`` | ``np.float16`` | ``np.float32`` | ``np.float64`` [Default = 1.0]
        Scaling factor to use for mapping points to image size that may have been scaled down for corner detection.

    Returns
    ---
    If corners were detected, returns color image with corner points drawn and point coordinates rescaled to original image coordinates by `scale` value, otherwise returns ``tuple`` of `None, None`.
    """
    if found:
        if len(color_img.shape) < 3:
            draw = cv.cvtColor(np.copy(color_img),cv.COLOR_GRAY2BGR) # assumes input was grayscale
        else:
            draw = np.copy(color_img)
        
        # Draw corners
        if type(scale) in [int, float, np.float16, np.float32, np.float64]:
            points = points * scale
            _ = cv.drawChessboardCorners(draw, size, points, found)
        
        elif type(scale) in [np.ndarray, tuple, list]:
            y_vals = points[:,0:1] * scale[0]
            x_vals = points[:,1:2] * scale[1]
            points = np.hstack((y_vals,x_vals))
            _ = cv.drawChessboardCorners(draw, size, points, found)
        
        # Make sure coordinates drop extraneous dimensions
        points = points.squeeze()
        
        return draw, points
    
    else:
        return None, None

def get_box_size(points:np.ndarray,
                 chckr_board_size:tuple|list|np.ndarray,
                 dim:str='h',
                 idx:int=None,
                 retr_all_sizes=False):
            """
            Usage
            ---
            Calculate average box size from checkerboard pattern corner points along single dimenion.

            Parameters
            ---
            points : ``np.ndarray``
                Array of checker board pattern points detected.
            
            chckr_board_size : ``tuple`` | ``list`` | ``np.ndarray``
                Size of checker board pattern with the format `(row, columns)`
            
            dim : ``str`` [Default = 'h']
                Dimension to calculate checkboard box sizes along, `h` for horizontal dimension (x-values) or `v` for vertical dimension (y-values).
            
            idx : ``int`` [Default = `None`]
                Index corresponding to dimension opposite of box size calculations. If not provided (recommended), uses default indices:
            
            | `dim` | `index` | aligns points |
            | :---: | :-----: | :-----------: |
            |  `h`  |    0    |    vertical   |
            |  `v`  |    1    |   horizontal  |

            retr_all_sizes : ``bool`` [Default = `False`]
                If the list of box sizes should be returned with the average size for that dimension, when `False` returns `None` instead

            Returns
            ---
            Tuple containing average checkboard pattern box size along dimension `dim` provided, and when `retr_all_sizes` is `True`, also returns a list containing all box sizes, otherwise returns `None`
            """

            # Check inputs
            assert dim is not None and isinstance(dim,str) and dim.lower() in ['h','v'], f"Must provide `dim` string pertaining to horizontal [h] or vertical [v] points."
            if idx is None:
                # NOTE `h` and `v` indices provide to the alignment vector, but output box dimensions in provided dimension
                # If `h` is given, then calculations are made by taking the differece between groups of vertical points for x-dimension box-size
                print(f"NOTE: using default indicies `h`:0 and `v`:1 since no value given for argument `idx`")

                indices = {
                        'h':0, #
                        'v':1 
                        }
                idx = indices[dim]

            # Board row and column counts
            row, col = chckr_board_size
            count = col if dim == 'v' else row

            # Sort points along dim idx
            sorted_points = points.squeeze()[points.squeeze()[:,idx].argsort()]
            
            # Calculate all box sizes in dimension `dim`
            prev = 0
            box_sizes = list()
            for e in range(count, len(sorted_points) + count, count):
                # Get all points on the same axis
                subset = sorted_points[prev:e]
                prev = e
                # 
                box_diffs = np.diff(subset[subset[:,0|(not idx)].argsort()][:,0|(not idx)])
                _ = [box_sizes.append(b) for b in box_diffs.tolist()]

            return (np.average(box_sizes), box_sizes) if retr_all_sizes else (np.average(box_sizes), None)

def check_imgsz(color_img:np.ndarray,
                size_limit:int=1440
                ):
    """
    Usage
    ---
    Checks if input image is over provided size limit, rescales image as needed, then outputs rescaled color and gray images with scaling factor.
    
    Parameters
    ---
    color_img : ``np.ndarray``
        Input image, should be color (3-channel) but will be converted to BGR-color image if input image has only single channel (assumed grayscale)

    size_limit : ``int`` [Default = 1440]
        Integer size limit for image height. Default value tested to be upper limit for OpenCV to detect chessboard pattern corners

    Returns
    ---
    In order resized color image, resized grayscale image, and scale factor as ``float``
    """

    # Scale ratio
    original_h, original_w, channels = color_img.shape if len(color_img.shape) == 3 else (*color_img.shape, None)
    scaling_factor = original_h / size_limit
    scaling_factor = scaling_factor if scaling_factor >= 1.0 else 1.0
    
    # Downscale since OpenCV has difficult time with detection on image heights greater than 1440
    if scaling_factor > 1.0:
        new_h, new_w = int(original_h / scaling_factor), int(original_w / scaling_factor)
        resize = (new_w, new_h)
    
    elif scaling_factor == 1.0:
        resize = (original_w, original_h)

    if channels is None:
        color_img = cv.cvtColor(np.copy(color_img), cv.COLOR_GRAY2BGR) # assumes input was grayscale

    process_img = cv.resize(np.copy(color_img), resize, interpolation=cv.INTER_CUBIC)
    process_gray = cv.cvtColor(np.copy(process_img), cv.COLOR_BGR2GRAY)

    return process_img, process_gray, scaling_factor

def sub_pixel_loc(gray_img:np.ndarray,
                      points:np.ndarray,
                      iterations:int = 50,
                      epsilon_acc:float = 1E-3,
                      half_window_sz:tuple|list = (3,3),
                      sub_pxl_conditions = cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER
                      ):

    """
    Usage
    ---
    Find sub-pixel accurate position of detected points for image.

    Parameters
    ---
    gray_img : ``np.ndarray``
        Input grayscale, single channel image.

    points : ``np.ndarray``
        Points to find sub-pixel accurate positions, points will be converted to ``np.float32`` data type.
    
    iterations : ``int`` [Default = 50]
        Number of iterations to use with count or max iteration termination conditions.
    
    epsilon_acc : ``float`` [Default = 0.001]
        The accuracy or change in parameters for termination conditions.

    half_window_sz : ``tuple`` | ``list`` [Default = (3, 3)]
        List or tuple with half-length dimension of the window to search in, see [OpenCV Docs](https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga354e0d7c86d0d9da75de9b9701a9a87e) for more information
    
    sub_pxl_conditions : ``int``
        See [OpenCV Docs](https://docs.opencv.org/4.x/d9/d5d/classcv_1_1TermCriteria.html) for additional information.
        
        `cv.TERM_CRITERIA_MAX_ITER == 1`
        
        `cv.TERM_CRITERIA_EPS == 2`
        
        `cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS == 3`

    Returns
    ---
    Array of sub pixel locations detected.

    """

    assert sub_pxl_conditions in range(1,4) and type(sub_pxl_conditions) == int, f"Subpixel conditions must be integer values between 1 and 3, see https://docs.opencv.org/4.x/d9/d5d/classcv_1_1TermCriteria.html for additional information"

    assert len(gray_img.shape) == 2, f"Input image is only allowed as single-channel grayscale image"
    iterations = int(iterations) if type(iterations) == float else iterations
    assert type(iterations) == int, f"Iterations must be an integer number"
    assert type(epsilon_acc) == float and epsilon_acc < 1 and epsilon_acc > 0, f"Invalid input for `epsilon_offset`, must be positive float value between 0 and 1."
    assert type(half_window_sz) in [list, tuple] and all([type(e) == int for e in half_window_sz]), f"Only tuple or list allowed for `half_window_sz` with integer entries, see https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga354e0d7c86d0d9da75de9b9701a9a87e for more information"

    if sub_pxl_conditions == 3:
        # All conditions
        sub_px_criteria = (sub_pxl_conditions, iterations, epsilon_acc)
    
    elif sub_pxl_conditions == 2:
        # Only epsilon accuracy condition
        sub_px_criteria = (sub_pxl_conditions, epsilon_acc)
    
    elif sub_pxl_conditions == 1:
        # Only max iteration or count condition
        sub_px_criteria = (sub_pxl_conditions, iterations)
    
    sub_px_locations = cv.cornerSubPix(np.copy(gray_img), points.astype(np.float32), half_window_sz, (-1,-1), sub_px_criteria)

    return sub_px_locations

def camera_cal_points(
        path_to_imgs:str|Path, # path of images
        filename_root:str='', # image naming convention
        board_size:tuple|list=(7,10), # number of inner checkerboard squares, (Row,Columns)
        BOX_ACTUAL_SIZE:int|float=1, # length of checkerboard squares in millimeters
        vizualize:bool=False, # show image after complete
        imgsz_lim:int=1440, # limit image size for opencv processing, downscaled for pattern detection
    ):
    """
    Usage
    ---

    Parameters
    ---
    path_to_imgs : ``str`` | ``pathlib.Path``
        Path to image files or full file path to image
    
    filename_root : ``str`` [Default = '']
        Common and continuous filename root string to search for using glob-like pattern; example: '*filename*' --> filename.png, filename-1137.jpeg, _0a_filename_1j_.tif
    
    board_size : ``tuple`` | ``list`` [Default = (7,10)]
        Count of inner-corners for checkerboard pattern `[points-per-row, points-per-column]`.
    
    BOX_ACTUAL_SIZE : ``int`` | ``float`` [Default = 1]
        Physical length of checkerboard square-sides measured in millimeters; default is 1 millimeter

    vizualize : ``bool`` [Default = False]
        Option to show image after processing.
    
    imgsz_lim : ``int`` [Default = 1440]
        Upper limit of image size height for OpenCV processing, downscaled for pattern detection; default height limit of 1440 determined in testing.

    Returns
    ---
    """
    
    # Create object points matrix
    object_pnts = np.zeros((np.prod(board_size),3), np.float32)
    object_pnts[:,:2] = np.mgrid[:board_size[0],:board_size[1]].T.reshape(-1,2)
    
    # Verify input
    assert type(BOX_ACTUAL_SIZE) in [int, float], f"Box edge dimension must be a either integer or float type, not {type(BOX_ACTUAL_SIZE)}" # default value is 1 mm
    assert type(imgsz_lim) == int, f"The value for `imgsz_lim` must be integer not {type(imgsz_lim)}" # default value is 1440
    assert type(board_size) in [tuple, list] and len(board_size) == 2 and all([type(e) == int for e in board_size]), f"Bad input for `board_size` argument, must use type tuple or list, with only two integer values."
    
    # Check path to images
    img_files = images_from_path(path_to_imgs, filename_root)

    # Outputs
    points = []
    x_sizes = []
    y_sizes = []
    object_points = []
    last_img = None

    # Process images
    for im in img_files:
        img = cv.imread(str(im))
        process_img, process_gray, scale_factor = check_imgsz(img)

        found_corners, corners = cv.findChessboardCornersSB(np.copy(process_gray), board_size, cv.CALIB_CB_EXHAUSTIVE + cv.CALIB_CB_ACCURACY)
    
        if vizualize and found_corners:
            # Visualize
            vis_corners, _ = visualize_corners(process_img, board_size, found_corners, corners.squeeze(), 1.0)
            show_im(vis_corners)
        
        elif found_corners and not vizualize:
            print(f"Found corners for {im.name}")
            object_points.append(object_pnts)
            
            sub_corners = sub_pixel_loc(process_gray, corners)
            points.append(sub_corners)
            
            # Find size of squares
            average_h_size, _ = get_box_size(sub_corners, board_size, 'h') # x-axis box size
            average_v_size, _ = get_box_size(sub_corners, board_size, 'v') # y-axis box size
            
            x_pxl = BOX_ACTUAL_SIZE / average_h_size # x-axis pixel size
            y_pxl = BOX_ACTUAL_SIZE / average_v_size # y-axis pixel size
            
            x_sizes.append(x_pxl)
            y_sizes.append(y_pxl)

            last_img = np.copy(img) # return the final image with detection

    return points, object_points, x_sizes, y_sizes, last_img

def new_cam_sizes():
    """Get checkerboard size from image after undistorting image"""
    pass

##-----##










