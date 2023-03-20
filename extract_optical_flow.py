import sys
sys.path.append('./RAFT/core')

import argparse
import os
import shutil
from os.path import join
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT 
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img

def load_image_list(image_files):
    images = []
    for imfile in sorted(image_files):
        images.append(load_image(imfile))
 
    images = torch.stack(images, dim=0)
    images = images.to(DEVICE)

    padder = InputPadder(images.shape)
    return padder.pad(images)[0]

def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()

def write_flo(filename, flow):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = flow.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    flow.tofile(f)
    f.close()

def viz(img1, img2, flo):
    img1 = img1[0].permute(1,2,0).cpu().numpy()
    img2 = img2[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img1, img2, flo], axis=0)
    
    cv2.imwrite('test_image.png', img_flo[:, :, [2,1,0]])

def run(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(DEVICE)
    model.eval()

    fwd_out_path = os.path.join(args.out_path, 'fwd_of')
    bwd_out_path = os.path.join(args.out_path, 'bwd_of')
    os.makedirs(fwd_out_path, exist_ok=True)
    os.makedirs(bwd_out_path, exist_ok=True)

    with torch.no_grad():

        images = glob.glob(os.path.join(args.frames_path, '*.jpg'))
        images = load_image_list(images)
        print(len(images))

        for i in range(images.shape[0]-1):
            img1 = images[i,None]
            img2 = images[i+1,None]

            flow_low, flow_up = model(img1, img2, iters=20, test_mode=True)
            #viz(img1, img2, flow_up)
            
            flow_path = os.path.join(fwd_out_path, '%05d_%05d.flo'%(i, i+1))
            write_flo(flow_path, flow_up[0].permute(1,2,0).cpu().numpy())

            flow_low, flow_up = model(img2, img1, iters=20, test_mode=True)
            
            flow_path = os.path.join(bwd_out_path, '%05d_%05d.flo'%(i, i+1))
            write_flo(flow_path, flow_up[0].permute(1,2,0).cpu().numpy())
            
if __name__=='__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_path', help="path to raw video frames")
    parser.add_argument('--out_path', help="results path")
    
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--out_dir', help="path to output directory")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    args = parser.parse_args()

    run(args)

