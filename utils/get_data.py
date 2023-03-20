import os
import sys
import shutil
import cv2
import numpy as np
import torch

def get_video_resolution(frames_path):
    video_files = os.listdir(frames_path)
    video_files.sort()
    file_path = os.path.join(frames_path, video_files[0])
    frame = cv2.imread(file_path)
    height = frame.shape[0]
    width = frame.shape[1]
    return height, width

def get_nr_video_frames(frames_path):
    video_files = os.listdir(frames_path)
    return len(video_files)

def get_video_frames_bgr(config, frames_path):
    height = config.getint('General', 'working_h')
    width = config.getint('General', 'working_w')

    video_files = os.listdir(frames_path)
    video_files.sort()
    frames = []
    for video_file_name in video_files:
        video_file_path = os.path.join(frames_path, video_file_name)
        frame = cv2.imread(video_file_path)
        frame = cv2.resize(src=frame,
                           dsize=(width, height),
                           interpolation=cv2.INTER_CUBIC)
        frame = frame / 255
        frames.append(torch.tensor(frame, dtype=torch.float32))
    return frames

def get_video_frames_rgb(config, frames_path):
    height = config.getint('General', 'working_h')
    width = config.getint('General', 'working_w')

    video_files = os.listdir(frames_path)
    video_files.sort()
    frames = []
    for video_file_name in video_files:
        video_file_path = os.path.join(frames_path, video_file_name)
        frame = cv2.imread(video_file_path)
        frame = cv2.resize(src=frame,
                           dsize=(width, height),
                           interpolation=cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame / 255
        frames.append(torch.tensor(frame, dtype=torch.float32))
    return frames
