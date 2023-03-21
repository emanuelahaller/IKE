"""
Main functions to generate initial soft-segmentation solutions
"""
import sys
import os
import torch
import cv2
import numpy as np

import ike_utils.get_data


def gauss_initialization_soft_segs(n_frames, height, width, sigma_percent=0.3):
    """
    Generate central Gaussian initial soft-segmentation maps  

    [in] n_frames       - number of frames 
    [in] height         - frame height 
    [in] width          - frame width
    [in] sigma_percent  - gaussian sigma will be sigma_percent * min(height, width)
    [out] soft_segs     - list of initial soft-segmentation maps 
    """
    sigma = min(height, width) * sigma_percent
    x = np.arange(0, width, 1)
    y = np.arange(0, height, 1)
    xx, yy = np.meshgrid(x, y)
    cx = np.int32(width * 0.5)
    cy = np.int32(height * 0.5)
    xx = xx - cx
    yy = yy - cy
    soft_seg = np.exp((-1) * (xx * xx + yy * yy) * (1 / (2 * sigma * sigma)))
    min_value = np.min(soft_seg)
    max_value = np.max(soft_seg)
    soft_seg = (soft_seg - min_value) / (max_value - min_value)
    soft_segs = []
    for _ in range(n_frames):
        soft_seg_ = np.zeros((height, width, 1))
        soft_seg_[:, :, 0] = soft_seg
        soft_segs.append(torch.tensor(soft_seg_, dtype=torch.float32))
    return soft_segs


def random_initialization_soft_segs(n_frames, height, width):
    """
    Generate random initial soft-segmentation maps 

    [in] n_frames       - number of frames 
    [in] height         - frame height 
    [in] width          - frame width
    [out] soft_segs     - list of initial soft-segmentation maps 
    """
    soft_segs = []
    for _ in range(n_frames):
        np.random.seed(115)
        soft_seg = np.random.rand(height, width, 1)
        soft_segs.append(torch.tensor(soft_seg, dtype=torch.float32))
    return soft_segs


def uniform_initialization_soft_segs(n_frames, height, width):
    """
    Generate random initial soft-segmentation maps

    [in] n_frames       - number of frames 
    [in] height         - frame height 
    [in] width          - frame width
    [out] soft_segs     - list of initial soft-segmentation maps 
    """
    soft_segs = []
    for _ in range(n_frames):
        soft_seg = np.ones((height, width, 1))
        soft_segs.append(torch.tensor(soft_seg, dtype=torch.float32))
    return soft_segs


def get_initialization(config, n_frames):
    """
        Get initialization for soft-segmentation labels 

        [in] config - configuration data
        [in] n_frames - number of frames for current video 
        [out] init_soft_segs - list of initial soft-segmentations 
    """
    seed_type = config.getint('Graph Module', 'seed_type')
    height = config.getint('General', 'working_h')
    width = config.getint('General', 'working_w')

    if seed_type == 0:
        return random_initialization_soft_segs(n_frames, height, width)
    elif seed_type == 1:
        return uniform_initialization_soft_segs(n_frames, height, width)
    elif seed_type == 2:
        if config.has_option('Graph Module', 'seed_type_sigma_percent'):
            sigma_percent = config.getfloat('Graph Module',
                                            'seed_type_sigma_percent')
            return gauss_initialization_soft_segs(n_frames, height, width,
                                                  sigma_percent)
        else:
            return gauss_initialization_soft_segs(n_frames, height, width)
    else:
        return AssertionError('Incorrect seed type')
