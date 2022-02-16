# This  module follows quite strictly the tutorial
# https://www.thepythoncode.com/article/extract-frames-from-videos-in-python

from datetime import timedelta
from plistlib import InvalidFileException
import cv2
import numpy as np
import os
from tqdm import tqdm


class VideoToFramesException(Exception):
    pass


ROOT_DIR = os.getcwd()
if os.path.basename(ROOT_DIR) != "lst-project":
    raise VideoToFramesException(
        f"Please run this module only in root directory lst-project. You're in {os.getcwd()}")
SAVING_FRAMES_PER_SECOND = 5
SOURCE_DIR = os.path.join(ROOT_DIR, "data", "video_data")
TARGET_DIR = os.path.join(ROOT_DIR, "data", "target")


def get_saving_frames_durations(cap, saving_fps):
    """A function that returns the list of durations where to save the frames"""
    s = []
    # get the clip duration by dividing number of frames by the number of frames per second
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    # use np.arange() to make floating-point steps
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s


def main(source_dir, target_dir):
    # make a folder by the name of the video file
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    if not os.path.isdir(source_dir):
        raise VideoToFramesException(
            f"Video data not found. Please ensure there's a folder of videos at {source_dir}")


    for filename in tqdm(os.listdir(source_dir)):
        video_file = os.path.join(source_dir, filename)

        if not os.path.isfile(video_file):
            raise InvalidFileException(f"Video file {video_file} is not a file.")
        # read the video file
        cap = cv2.VideoCapture(video_file)

        # get the FPS of the video
        fps = cap.get(cv2.CAP_PROP_FPS)

        # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
        saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)

        # get the list of duration spots to save
        saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)

        
        count = 0
        is_read, frame = cap.read()
        video_name = os.path.basename(video_file).split(".")[0]
        while is_read:
            # get the duration by dividing the frame count by the FPS
            frame_duration = count / fps
            try:
                # get the earliest duration to save
                closest_duration = saving_frames_durations[0]
            except IndexError:
                # the list is empty, all duration frames were saved
                break
            if frame_duration >= closest_duration:
                # if closest duration is less than or equals the frame duration, 
                # then save the frame
                im_path = os.path.join(target_dir, video_name + "_" + str(count) + ".png")
                write_successful = cv2.imwrite(im_path, frame) 
                if not write_successful:
                    raise VideoToFramesException(f"Couldn't write path: {im_path}")
                # drop the duration spot from the list, since this duration spot is already saved
                try:
                    saving_frames_durations.pop(0)
                except IndexError:
                    pass
            # increment the frame count
            count += 1
            is_read, frame = cap.read()

if __name__ == "__main__":
    main(SOURCE_DIR, TARGET_DIR)
