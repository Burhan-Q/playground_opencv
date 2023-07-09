"""
video_utils
Author: Burhan Qaddoumi
Source: github.com/Burhan-Q
Date: 2023-05-26
"""

import cv2 as cv
import numpy as np
import pafy # URL conversion readable by OpenCV
from pathlib import Path
import re
import sys
import yaml
import datetime as dt

VID_FORMATS = {'H264':'.mp4','MP4V':'.mp4'} # https://learn.microsoft.com/en-us/windows/win32/medfound/video-fourccs
VID_PROPS = {n[9:]:getattr(cv,n) for n in dir(cv) if n.startswith('CAP_PROP_')}
VID_BACKENDS = {getattr(cv,n):n[4:] for n in dir(cv) if n.startswith('CAP_') and not n.startswith('CAP_PROP_')}
SETTINGS_3DOCAM = {'max'  : [2160, 3840, 30],
                   'high' : [1440, 2560, 60],    # NOTE double check this works
                   'highL': [1440, 2560, 30],
                   #'med'  : [1080, 1920, 60],    # NOTE this does not seem to work
                   #'medL' : [1080, 1920, 25],    # NOTE this does not seem to work
                   'low'  : [ 720, 1268, 60]     # maybe FPS can be higher, need to test
                   }

# matches 'youtu.be/*', 'youtube.com/*', 'youtube.com/*', & 'youtube.com/watch?v=*' appended with any prefix: 'http://', 'https://', '(https|http)://www.' or no-prefix
YT_FORMAT = r'(http|https)*(\:\/\/)*(www\.)*youtu(be)*\.*(be|com)\/[a-zA-Z0-9]+\?*(v\=)*[a-zA-Z0-9]+'

def check_deviceID(index:int,pos_only=False):
    """Verify device index is integer, check if positive and non-zero when `pos_only=True`"""
    index = int(index) if type(index) == str and index.isnumeric() else index

    if pos_only:
        assert index > 0, f"Value must be positive, non-zero integer, not {index}"
    
    return index

def check_API_backend(backend:int,override:bool=False):
    """Check that API backend is valid and correct for platform; bypass with `override=True`"""
    assert backend in VID_BACKENDS.keys(), f"Unsupported video backend API value {backend}."

    # Only use supported backend if Windows
    backend = cv.CAP_DSHOW if backend not in [700, 1400] and not override else backend
    backend = backend if (sys.platform == 'win32' and backend in [700, 1400]) or override else cv.CAP_DSHOW

    return backend

def check_cam_settings(settings:dict|list|tuple,
                       defaults:dict=None
                       ):
    """Verify that settings are valid for camera, if no camera defaults provided, uses `SETTINGS_3DOCAM` as defaults."""
    keys = [
            cv.CAP_PROP_FRAME_HEIGHT,
            cv.CAP_PROP_FRAME_WIDTH,
            cv.CAP_PROP_FPS,
            ]
    
    # Check settings for dictionary input
    if isinstance(settings,dict):
        assert all([k in settings.keys() for k in keys]), f"Not all keys present"
        setting_list = np.roll(sorted([int(settings[k]) for k in keys]),-1).tolist()
    
    # Check settings for other iterable input
    elif isinstance(settings,(list,tuple)):
        setting_list = np.roll(sorted(settings),-1).astype(int).tolist() # ensure in correct order, H, W, FPS

    else:
        assert False, f"Settings type {type(settings)} not valid."

    if defaults is None:
        assert setting_list in SETTINGS_3DOCAM.values(), f""
    
    elif defaults is not None:
        list_defaults = defaults.values() if len(defaults.values()) > 1 else [defaults.values(),]
        assert setting_list in list_defaults, f""
    
    # return settings # NOTE may not really need to return anything

def test_vidpath(path:str,format:str):
    """Verify that video save parent path exists and uses correct file extension"""
    input_path = Path(path)
    
    assert input_path.parent.exists(), f"Parent of input path {str(input_path.parent)} does not exist."
    assert format in VID_FORMATS.values(), f"Ensure format is compatible with this function."
    
    return path

def local_yt(link:str,
             save_path:str,
             write_type:str='MP4V',
             file_name:str=None
             ):

    # YouTube video link
    assert re.search(YT_FORMAT,link) is not None, f"Link provided {link} is not a known format, try again"
    source_link = link
    convrt_link = pafy.new(source_link).getbestvideo(preftype='mp4').url # will only get MP4 version with best quality

    # Output file name (full path)
    file_ext = VID_FORMATS[write_type]
    file_name = file_name if file_name is not None and type(file_name) == str else 'YT_video'
    save_path = '/'.join([save_path.removesuffix('/'), f'{file_name}{file_ext}']) if Path(save_path).is_dir() else str(Path(save_path).with_suffix(file_ext))
    save_path = test_vidpath(save_path, file_ext)

    # Video stream
    cap = cv.VideoCapture(convrt_link)

    # Video information
    w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv.CAP_PROP_FPS)
    w, h = int(w), int(h)

    if w == h:
        print(f"NOTE: video resolution is {w} x {h}")

    assert type(w) == int and type(h) == int, f"Width and height must be integers"

    # Output writer
    out = cv.VideoWriter(save_path, cv.VideoWriter_fourcc(*write_type), fps, (w,h))

    # Write output
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No frame, exiting")
            break
        out.write(frame)

    # Release resources
    cap.release()
    out.release()

def cam_image_settings(cam_indx:int, # integer index of camera to open
                       vHeight:int, # image pixel height setting
                       vWidth:int, # image pixel width setting
                       vFPS:int, # frames per second setting
                       vBackEnd:int, # video API backend, see OpenCV docs
                       cam_obj:cv.VideoCapture=None, # existing VideoCapture object; [Default = `None`]
                       ) -> cv.VideoCapture:
    
    # Construct settings dictionary
    settings = {
        cv.CAP_PROP_FRAME_HEIGHT:int(vHeight),
        cv.CAP_PROP_FRAME_WIDTH :int(vWidth),
        cv.CAP_PROP_FPS         :int(vFPS),
        cv.CAP_PROP_BACKEND     :int(vBackEnd)
        }
    
    # Verify camera
    cam_indx = check_deviceID(cam_indx)
    assert isinstance(cam_indx,(int,float)) or type(cam_obj) == cv.VideoCapture, f"Must input integer index for camera OR valid OpenCV VideoCapture object."
    # Verify settings
    _ = check_cam_settings(settings)
    vBackEnd = check_API_backend(vBackEnd)

    # Create new VideoCapture object
    if cam_obj is None:
        cam_obj = cv.VideoCapture(cam_indx,int(vBackEnd))
    
    # Open VideoCapture object
    elif not cam_obj.isOpened():
        cam_obj = cam_obj.open(cam_indx,int(vBackEnd))
    
    # Apply settings to VideoCapture object
    for k,v in settings.items():
        if cam_obj.get(k) != v and k != cv.CAP_PROP_BACKEND:
            cam_obj.set(k,v)
    
    if not cam_obj.isOpened():
        # TODO LOG ERROR
        pass

    return cam_obj

def load_cam_stream(vDevice:int,
                    vHeight:int,
                    vWidth:int,
                    vFPS:int,
                    vBackEnd:int
                    ) -> cv.VideoCapture:
    
    # Construct settings dictionary
    settings = {
        cv.CAP_PROP_FRAME_HEIGHT:int(vHeight),
        cv.CAP_PROP_FRAME_WIDTH :int(vWidth),
        cv.CAP_PROP_FPS         :int(vFPS),
        cv.CAP_PROP_BACKEND     :int(vBackEnd)
        }
    
    # Verify inputs
    vDevice = check_deviceID(vDevice)
    assert isinstance(vDevice,(int,float)), f"Must input integer index for camera."
    vBackEnd = check_API_backend(vBackEnd)
    _ = check_cam_settings(settings)
    
    # Create VideoCapture object using device
    cap = cv.VideoCapture(vDevice, vBackEnd)
    
    # Apply settings
    if cap.isOpened():
        for k,v in settings.items():
            if cap.get(k) != v and k != cv.CAP_PROP_BACKEND:
                cap.set(k,v)
    
    elif not cap.isOpened():
        cap = cap.open(vDevice)
        
        if not cap.isOpened():
            # TODO add error logging
            print("Unable to open camera")
            exit()
    
    else:
        # TODO add error logging
        pass

    return cap

def get_cam_props(cam_obj:cv.VideoCapture,
                  save_path:str|Path
                  ):
    """Save copy of camera properties as YAML file."""

    if not cam_obj.isOpened():
        # TODO add error logging
        no_cam = f"Input cv.VideoCapture object is not open."
        return

    # Setup save path
    date = dt.datetime.today().date().isoformat()
    in_path = str(save_path).replace('\\','/')
    save_path = Path(in_path) if not Path(in_path).is_dir() else Path(in_path).parent
    save_file = Path(Path.home().as_posix() + f'/{date}_camera_settings.yaml')
    
    ## Not able to find save path
    if not save_path.exists():
        #TODO add logging entry
        # date = dt.datetime.today().date().isoformat()
        # save_file = Path(Path.home().as_posix() + f'/{date}_camera_settings.yaml')
        msg = f"Unable to find {in_path}, saving settings to {save_file} instead" # LOG message
    
        ## Existing save file, check to overwrite
        if save_file.exists():
            prompt = f"Found existing file {save_file.as_posix()}, overwrite this file ('Y'/'N')? \n"
            overwrite = None
            while overwrite is None:
                overwrite = input(prompt)
                overwrite = None if len(overwrite) < 1 or overwrite.upper()[0] not in ['Y','N'] else overwrite.upper()[0]
            
            if overwrite == 'N':
                exit() # exit python, could return None instead, should figure out preference
    
    # Collect information from camera stream
    prop_dict = dict()
    try:
        [prop_dict.update({prop:cam_obj.get(idnum)}) for prop, idnum in VID_PROPS.items() if cam_obj.get(idnum) != -1.0]
    
        ## Convert information
        prop_dict['FOURCC'] = int(prop_dict['FOURCC']).to_bytes(4,sys.byteorder).decode()
        prop_dict['BACKEND'] = VID_BACKENDS[prop_dict['BACKEND']]
    
    except:
        # TODO Error logging message here
        pass

    # TODO Add camera calibration to `prop_dict` before saving to YAML file

    # Output to file
    with open(save_file,'w') as y:
        _ = yaml.safe_dump(prop_dict, y)

cap = load_cam_stream(0,1080,1920,30,cv.CAP_DSHOW)