"""
video_utils
Author: Burhan Qaddoumi
Source: github.com/Burhan-Q
Date: 2023-05-26
"""

import cv2 as cv
import pafy # URL conversion readable by OpenCV
from pathlib import Path
import re

VID_FORMATS = {'H264':'.mp4','MP4V':'.mp4'} # https://learn.microsoft.com/en-us/windows/win32/medfound/video-fourccs

# matches 'youtu.be/*', 'youtube.com/*', 'youtube.com/*', & 'youtube.com/watch?v=*' appended with any prefix: 'http://', 'https://', '(https|http)://www.' or no-prefix
YT_FORMAT = r'(http|https)*(\:\/\/)*(www\.)*youtu(be)*\.*(be|com)\/[a-zA-Z0-9]+\?*(v\=)*[a-zA-Z0-9]+'

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
