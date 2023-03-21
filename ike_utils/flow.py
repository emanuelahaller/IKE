import os
import sys
import shutil
import torch
import numpy as np

def read_optical_flow_file(file_path):
    """
    Read optical flow file '.flo'

    [in] file_path  - path of the optical flow file
    [out] data2D    - optical flow data - height x width x 2 (dx & dy displacements)
    """
    with open(file_path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
        else:
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2 * w * h)
            data2D = np.resize(data, (h, w, 2))
    return data2D


def preprocess_flow_data(flow, working_h, working_w):
    """
    Preprocess flow data - rescale 

    [in] flow       - optical flow data - height x width x 2 (dx & dy displacements)
    [in] working_h  - height of the final flow map 
    [in] working_w  - width of the final flow map
    [out] flow      - optical flow data - working_height x working_width x 2 (dx & dy displacements)
    """
    flow_h = flow.shape[0]
    flow_w = flow.shape[1]

    flow = torch.tensor(flow, dtype=torch.float32)
    flow = flow.unsqueeze(0)
    flow = flow.permute(0, 3, 1, 2)
    flow = torch.nn.functional.interpolate(flow,
                                           size=(working_h, working_w),
                                           mode='bilinear',
                                           align_corners=False)
    flow = flow.squeeze()
    flow = flow.permute(1, 2, 0)

    flow[:, :, 0] = flow[:, :, 0] * (working_w / flow_w)
    flow[:, :, 1] = flow[:, :, 1] * (working_h / flow_h)

    return flow

def get_video_optical_flow(flows_path, n_frames, height, width):
    """
        Read precomputed optical flow for a movie

        Optical flow files will repect the following naming convention:
        [first_frame_idx]_[second_frame_idx].flo

        [in] flows_path     - main path for flow data
        [in] n_frames       - number of frames of the considered video
        [in] height         - working height
        [in] width          - working width
    """
    optical_flows = []
    for idx in range(n_frames - 1):
        name = '%05d_%05d.flo' % (idx, idx + 1)

        flow = read_optical_flow_file(
            os.path.join(flows_path, name))
        flow = preprocess_flow_data(flow, height, width)
        flow = flow.cuda()
        optical_flows.append(flow)

    return optical_flows
