import os
import sys
import shutil
import numpy as np
import torch
import cv2


def save_soft_segs(out_path, soft_segs, height, width):
    """
        Save soft segs at original scale 

        [in] out_path - output path 
        [in] soft_segs - soft-segmentation maps 
        [in] height - original frame height 
        [in] width - original frame width
    """
    os.makedirs(out_path, exist_ok=True)
    for frame_idx, soft_seg in enumerate(soft_segs):
        obj_soft_seg = soft_seg.cuda()
        obj_soft_seg = obj_soft_seg.permute(2, 0, 1)
        obj_soft_seg = obj_soft_seg.unsqueeze(0)
        obj_soft_seg = torch.nn.functional.interpolate(obj_soft_seg,
                                                       size=(height, width),
                                                       mode='bilinear',
                                                       align_corners=False)
        obj_soft_seg = obj_soft_seg.squeeze(0)
        obj_soft_seg = obj_soft_seg.permute(1, 2, 0)
        torch.save(obj_soft_seg, os.path.join(out_path,
                                              '%05d.pt' % (frame_idx)))


def save_soft_segs_working_size(out_path, soft_segs):
    os.makedirs(out_path, exist_ok=True)
    for frame_idx, soft_seg in enumerate(soft_segs):
        torch.save(soft_seg, os.path.join(out_path, '%05d.pt' % (frame_idx)))


def save_features_working_size(out_path, soft_segs):
    os.makedirs(out_path, exist_ok=True)
    for frame_idx, soft_seg in enumerate(soft_segs):
        torch.save(soft_seg, os.path.join(out_path, '%05d.pt' % (frame_idx)))


def save_soft_segs_images(out_path, soft_segs, height, width):
    """
        Save soft segs at original scale - image format

        [in] out_path - output path 
        [in] soft_segs - soft-segmentation maps 
        [in] height - original frame height 
        [in] width - original frame width
    """
    os.makedirs(out_path, exist_ok=True)
    for frame_idx, soft_seg in enumerate(soft_segs):
        obj_soft_seg = soft_seg.cuda()
        obj_soft_seg = obj_soft_seg.permute(2, 0, 1)
        obj_soft_seg = obj_soft_seg.unsqueeze(0)
        obj_soft_seg = torch.nn.functional.interpolate(obj_soft_seg,
                                                       size=(height, width),
                                                       mode='bilinear',
                                                       align_corners=False)
        obj_soft_seg = obj_soft_seg.squeeze(0)
        obj_soft_seg = obj_soft_seg.permute(1, 2, 0)
        obj_soft_seg = obj_soft_seg.cpu().numpy()
        cv2.imwrite(os.path.join(out_path, '%05d.png'%(frame_idx)), np.uint8(obj_soft_seg*255))
       