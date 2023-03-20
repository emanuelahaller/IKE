import os
import sys
import shutil
import cv2
import math
import torch
import numpy as np


def compute_rounded_flow(flow_x, flow_y, prev_flow_x, prev_flow_y, xx, yy,
                         working_h, working_w):
    new_flow_x = prev_flow_x + xx
    new_flow_y = prev_flow_y + yy

    x_0 = (torch.round(new_flow_x)).long()
    y_0 = (torch.round(new_flow_y)).long()

    x_0[x_0 >= working_w] = working_w - 1
    x_0[x_0 < 0] = 0

    y_0[y_0 >= working_h] = working_h - 1
    y_0[y_0 < 0] = 0

    final_flow_x = flow_x[y_0, x_0]
    final_flow_y = flow_y[y_0, x_0]

    del x_0
    del y_0

    return final_flow_x, final_flow_y


def precompute_chains_info_aux(flows, n_frames, working_h, working_w,
                               pivot_idx, xx, yy, offset, indexes,
                               computed_chains_x_dir, computed_chains_y_dir,
                               computed_chains_x_pos, computed_chains_y_pos):
    """
    Auxiliary function for computing chains info - for a given pivot frame and moving direction (fwd or bwd)

    [in] flows - list of flow info (fwd or bwd flow)
    [in] n_frames - number of frames in current video
    [in] pivot_idx - index of the pivot frame
    [in] xx - meshgrid with x coords
    [in] yy - meshgrid with y coords
    [in] offset - placement offset
    [in] indexes - indexes of considered destination frames
    [in/out] computed_chains_x_dir
    [in/out] computed_chains_y_dir
    [in/out] computed_chains_x_pos
    [in/out] computed_chains_y_pos
    """
    prev_flow_x = torch.zeros(working_h, working_w, dtype=torch.float32)
    prev_flow_y = torch.zeros(working_h, working_w, dtype=torch.float32)
    prev_flow_x = prev_flow_x.cuda()
    prev_flow_y = prev_flow_y.cuda()
    for idx in indexes:
        flow = flows[idx]
        flow_x = flow[:, :, 0]
        flow_y = flow[:, :, 1]

        interp_flow_x, interp_flow_y = compute_rounded_flow(
            flow_x, flow_y, prev_flow_x, prev_flow_y, xx, yy, working_h,
            working_w)

        del flow_x
        del flow_y

        prev_flow_x.add_(interp_flow_x)
        prev_flow_y.add_(interp_flow_y)

        computed_chains_x_dir[idx + offset] = (torch.reshape(
            interp_flow_x, (working_h * working_w, 1))).cpu()
        computed_chains_y_dir[idx + offset] = (torch.reshape(
            interp_flow_y, (working_h * working_w, 1))).cpu()

        del interp_flow_x
        del interp_flow_y

        x_coords = xx + prev_flow_x
        y_coords = yy + prev_flow_y

        computed_chains_x_pos[idx + offset] = (torch.reshape(
            x_coords, (working_h * working_w, 1)))
        computed_chains_y_pos[idx + offset] = (torch.reshape(
            y_coords, (working_h * working_w, 1)))

    del prev_flow_x
    del prev_flow_y

    return computed_chains_x_dir, computed_chains_y_dir, computed_chains_x_pos, computed_chains_y_pos


def precompute_chains_info_get_dir_features(computed_chains_x_dir,
                                            computed_chains_y_dir, pivot_idx,
                                            n_frames, half_window_size):
    indexes = np.concatenate(
        (np.arange(pivot_idx - half_window_size, pivot_idx, 1),
         np.arange(pivot_idx + 1, pivot_idx + half_window_size + 1, 1)))

    indexes[indexes < 0] = 0

    indexes[indexes >= n_frames] = n_frames - 1

    all_chains_x = np.concatenate([computed_chains_x_dir[i] for i in indexes],
                                  axis=1)
    all_chains_y = np.concatenate([computed_chains_y_dir[i] for i in indexes],
                                  axis=1)
    all_chains = np.concatenate((all_chains_x, all_chains_y), axis=1)

    all_chains = torch.tensor(all_chains, dtype=torch.float32)

    return all_chains


def compute_rounded_features(x_coords, y_coords, features, working_h,
                             working_w):
    x_0 = x_coords.long()
    y_0 = y_coords.long()

    bm_00 = (x_0 >= 0) & (x_0 < working_w) & (y_0 >= 0) & (y_0 < working_h)
    bm_00 = ~bm_00
    x_0[x_0 >= working_w] = working_w - 1
    x_0[x_0 < 0] = 0
    y_0[y_0 >= working_h] = working_h - 1
    y_0[y_0 < 0] = 0

    coords = x_0 + working_w * y_0
    new_features = features[coords[:, 0], :]

    pos_00 = torch.where(bm_00 == True)[0]
    new_features[pos_00, :] = 0

    return new_features


def precompute_chains_info_get_features(computed_chains_x_pos,
                                        computed_chains_y_pos, features,
                                        pivot_idx, n_frames, half_window_size,
                                        working_h, working_w):
    n_features = features[0].shape[1]
    descriptor_size = n_features * (2 * half_window_size + 1)
    frame_features = torch.zeros((working_h * working_w, descriptor_size))
    indexes = np.arange(pivot_idx - half_window_size,
                        pivot_idx + half_window_size + 1, 1)
    indexes[indexes < 0] = 0
    indexes[indexes >= n_frames] = n_frames - 1

    for idx in range(len(indexes)):
        x_coords = computed_chains_x_pos[indexes[idx]]
        y_coords = computed_chains_y_pos[indexes[idx]]

        frame_feat = features[indexes[idx]]
        interp_frame_feat = compute_rounded_features(x_coords, y_coords,
                                                     frame_feat, working_h,
                                                     working_w)

        frame_features[:, idx * n_features:(idx + 1) *
                       n_features] = interp_frame_feat
    return frame_features


def compute_neighbour_weights(dx_0, dy_0, dx_1, dy_1):
    dx_0 = dx_0**2
    dy_0 = dy_0**2
    dx_1 = dx_1**2
    dy_1 = dy_1**2

    d_00 = torch.sqrt(dy_0 + dx_0)
    d_01 = torch.sqrt(dy_0 + dx_1)
    d_10 = torch.sqrt(dy_1 + dx_0)
    d_11 = torch.sqrt(dy_1 + dx_1)

    del dx_0
    del dx_1
    del dy_0
    del dy_1

    d_00 = torch.exp(-d_00)
    d_01 = torch.exp(-d_01)
    d_10 = torch.exp(-d_10)
    d_11 = torch.exp(-d_11)

    sum_ = d_00 + d_01 + d_10 + d_11

    d_00 = d_00 / sum_
    d_01 = d_01 / sum_
    d_10 = d_10 / sum_
    d_11 = d_11 / sum_

    del sum_

    return d_00, d_01, d_10, d_11


def graph_compute_projection(gauss_weight, x_flow_coords, y_flow_coords,
                             working_h, working_w):
    x_0 = torch.floor(x_flow_coords)
    x_1 = torch.ceil(x_flow_coords)
    y_0 = torch.floor(y_flow_coords)
    y_1 = torch.ceil(y_flow_coords)

    w_00, w_01, w_10, w_11 = compute_neighbour_weights(x_flow_coords - x_0,
                                                       y_flow_coords - y_0,
                                                       x_flow_coords - x_1,
                                                       y_flow_coords - y_1)
    w_00 = 1
    w_01 = 1
    w_10 = 1
    w_11 = 1

    x_0 = x_0.long()
    y_0 = y_0.long()
    x_1 = x_1.long()
    y_1 = y_1.long()

    bm_00 = ((x_0 >= 0) & (x_0 < working_w) & (y_0 >= 0) &
             (y_0 < working_h)).float()
    bm_01 = ((x_1 >= 0) & (x_1 < working_w) & (y_0 >= 0) &
             (y_0 < working_h)).float()
    bm_10 = ((x_0 >= 0) & (x_0 < working_w) & (y_1 >= 0) &
             (y_1 < working_h)).float()
    bm_11 = ((x_1 >= 0) & (x_1 < working_w) & (y_1 >= 0) &
             (y_1 < working_h)).float()

    coords_00 = ((y_0 * working_w + x_0) * bm_00).long()
    coords_01 = ((y_0 * working_w + x_1) * bm_01).long()
    coords_10 = ((y_1 * working_w + x_0) * bm_10).long()
    coords_11 = ((y_1 * working_w + x_1) * bm_11).long()

    del x_0
    del x_1
    del y_0
    del y_1

    cues_00 = bm_00 * w_00 * gauss_weight
    cues_01 = bm_01 * w_01 * gauss_weight
    cues_10 = bm_10 * w_10 * gauss_weight
    cues_11 = bm_11 * w_11 * gauss_weight

    edge_weights = torch.cat((cues_00, cues_01, cues_10, cues_11), 1)
    coords = torch.cat((coords_00, coords_01, coords_10, coords_11), 1)

    return edge_weights, coords


def graph_perform_projection_undirected(pivot_soft_seg, dst_soft_seg,
                                        edge_weights, coords, working_h,
                                        working_w):
    rec_pivot_soft_seg = torch.zeros(working_h * working_w,
                                     1,
                                     dtype=torch.float32)
    rec_dst_soft_seg = torch.zeros(working_h * working_w,
                                   1,
                                   dtype=torch.float32)
    rec_pivot_soft_seg = rec_pivot_soft_seg.cuda()
    rec_dst_soft_seg = rec_dst_soft_seg.cuda()

    rec_pivot_soft_seg = rec_pivot_soft_seg.squeeze()

    values = pivot_soft_seg * edge_weights[:, 0, None]
    rec_pivot_soft_seg.index_add_(0, coords[:, 0], values[:, 0])

    values = dst_soft_seg  #* edge_weights[:,0,None]
    rec_dst_soft_seg = rec_dst_soft_seg + values[
        coords[:, 0]] * edge_weights[:, 0, None]

    values = pivot_soft_seg * edge_weights[:, 1, None]
    rec_pivot_soft_seg.index_add_(0, coords[:, 1], values[:, 0])

    values = dst_soft_seg  #* edge_weights[:,1,None]
    rec_dst_soft_seg = rec_dst_soft_seg + values[
        coords[:, 1]] * edge_weights[:, 1, None]

    values = pivot_soft_seg * edge_weights[:, 2, None]
    rec_pivot_soft_seg.index_add_(0, coords[:, 2], values[:, 0])

    values = dst_soft_seg  #* edge_weights[:,2,None]
    rec_dst_soft_seg = rec_dst_soft_seg + values[
        coords[:, 2]] * edge_weights[:, 2, None]

    values = pivot_soft_seg * edge_weights[:, 3, None]
    rec_pivot_soft_seg.index_add_(0, coords[:, 3], values[:, 0])

    values = dst_soft_seg  #* edge_weights[:,3,None]
    rec_dst_soft_seg = rec_dst_soft_seg + values[
        coords[:, 3]] * edge_weights[:, 3, None]

    rec_pivot_soft_seg = rec_pivot_soft_seg.unsqueeze(1)
    return rec_pivot_soft_seg, rec_dst_soft_seg