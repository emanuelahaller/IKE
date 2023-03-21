import os
import sys
import shutil
import torch
import cv2
import math
import time

import numpy as np
from torch import nn

import graph_module.utils
import ike_utils.save_data


class SpaceTimeGraph:
    def __init__(self, config, frames, fwd_flows, bwd_flows, object_cues):
        self.config = config

        self.n_frames = len(frames)
     
        self.object_cues = object_cues
        if len(self.object_cues) == 0:
            self.n_obj_cue_channels = 0
        else:
            self.n_obj_cue_channels = object_cues[0].shape[2]

        self.all_coords = [[''] * self.n_frames
                           for idx in range(self.n_frames)]
        self.all_edge_weights = [[''] * self.n_frames
                                 for idx in range(self.n_frames)]
        self.all_frame_features = 0

        self.n_graph_features = 0
        self.set_n_features()

        sigma = config.getfloat('Graph Module', 'SIGMA')
        self.compute_graph(fwd_flows, bwd_flows, frames, sigma)
        del fwd_flows
        del bwd_flows

        self.normalize_dir_features()
        self.normalize_features()

        self.fwd_bms = []
        self.bwd_bms = []

    def compute_graph(self, fwd_flows, bwd_flows, frames, sigma):

        working_h = self.config.getint('General', 'working_h')
        working_w = self.config.getint('General', 'working_w')

        if self.n_obj_cue_channels > 0:
            for frame_idx in range(self.n_frames):
                frame_object_cues = self.object_cues[frame_idx]
                frame_object_cues = torch.reshape(
                    frame_object_cues,
                    (working_h * working_w, self.n_obj_cue_channels))
                self.object_cues[frame_idx] = frame_object_cues

        for frame_idx in range(self.n_frames):
            frame = frames[frame_idx]
            frame = torch.reshape(frame, (working_h * working_w, 3))
            frames[frame_idx] = frame

        yy, xx = torch.meshgrid([
            torch.arange(0, working_h, dtype=torch.float32),
            torch.arange(0, working_w, dtype=torch.float32)
        ])
        xx = xx.cuda()
        yy = yy.cuda()

        gg = torch.arange(0, 16, 1, dtype=torch.float32).cuda()
        gg = torch.exp((-0.5) * ((gg) / sigma)**2)

        self.all_frame_features = torch.zeros(self.n_frames,
                                              working_h * working_w,
                                              self.n_graph_features)
        for pivot_idx in range(self.n_frames):
            computed_chains_x_pos = [''] * self.n_frames
            computed_chains_y_pos = [''] * self.n_frames

            computed_chains_x_dir = [''] * self.n_frames
            computed_chains_y_dir = [''] * self.n_frames

            computed_chains_x_dir[pivot_idx] = torch.zeros(
                working_h * working_w, 1)
            computed_chains_y_dir[pivot_idx] = torch.zeros(
                working_h * working_w, 1)
            computed_chains_x_pos[pivot_idx] = torch.reshape(
                xx, (working_h * working_w, 1))
            computed_chains_y_pos[pivot_idx] = torch.reshape(
                yy, (working_h * working_w, 1))

            # move fwd from current frame (pivot frame) - only 15 frames
            #indexes = np.arange(pivot_idx, n_frames-1, 1)
            indexes = np.arange(pivot_idx,
                                min(pivot_idx + 15, self.n_frames - 1), 1)
            computed_chains_x_dir, computed_chains_y_dir,\
                computed_chains_x_pos, computed_chains_y_pos = graph_module.utils.precompute_chains_info_aux(fwd_flows, self.n_frames, working_h, working_w, pivot_idx, xx, yy, 1, indexes,\
                    computed_chains_x_dir, computed_chains_y_dir, computed_chains_x_pos, computed_chains_y_pos)

            # move bwd from current frame (pivot frame) - only 15 frames
            #indexes = np.arange(pivot_idx-1, -1, -1)
            indexes = np.arange(pivot_idx - 1, max(-1, pivot_idx - 16), -1)
            computed_chains_x_dir, computed_chains_y_dir,\
                computed_chains_x_pos, computed_chains_y_pos = graph_module.utils.precompute_chains_info_aux(bwd_flows, self.n_frames, working_h, working_w, pivot_idx, xx, yy, 0, indexes,\
                    computed_chains_x_dir, computed_chains_y_dir, computed_chains_x_pos, computed_chains_y_pos)

            # extract direction features
            frame_dir_features = graph_module.utils.precompute_chains_info_get_dir_features(\
                computed_chains_x_dir, computed_chains_y_dir, pivot_idx, self.n_frames, self.config.getint('Graph Module', 'dir_features_half_chain_size'))
            frame_features = frame_dir_features

            del computed_chains_x_dir
            del computed_chains_y_dir

            features_half_chain_size = self.config.getint(
                'Graph Module', 'features_half_chain_size')
            if features_half_chain_size >= 0:
                cue_features = graph_module.utils.precompute_chains_info_get_features(\
                    computed_chains_x_pos, computed_chains_y_pos, self.object_cues, pivot_idx, self.n_frames, features_half_chain_size, working_h, working_w)
                frame_features = torch.cat((frame_features, cue_features), 1)

            self.all_frame_features[pivot_idx, :, :] = frame_features

            self.all_edge_weights[pivot_idx][pivot_idx] = gg[0]

            indexes1 = np.arange(pivot_idx,
                                 min(pivot_idx + 15, self.n_frames - 1), 1)
            #indexes1 = np.arange(pivot_idx, n_frames-1, 1)
            offsets1 = np.ones(indexes1.shape, dtype=np.int32)
            indexes2 = np.arange(pivot_idx - 1, max(-1, pivot_idx - 16), -1)
            #indexes2 = np.arange(pivot_idx-1, -1, -1)
            offsets2 = np.zeros(indexes2.shape, dtype=np.int32)
            indexes = np.concatenate((indexes1, indexes2))
            offsets = np.concatenate((offsets1, offsets2))

            for idx, offset in zip(indexes, offsets):

                x_coords = computed_chains_x_pos[idx + offset]
                y_coords = computed_chains_y_pos[idx + offset]

                edge_weights, coords =\
                    graph_module.utils.graph_compute_projection(gg[abs(idx+offset-pivot_idx)], x_coords, y_coords, working_h, working_w)

                self.all_edge_weights[pivot_idx][idx +
                                                 offset] = edge_weights.cpu()
                self.all_coords[pivot_idx][idx + offset] = coords.cpu()

    def set_n_features(self):
        dir_features_half_chain_size = self.config.getint(
            'Graph Module', 'dir_features_half_chain_size')
        features_half_chain_size = self.config.getint(
            'Graph Module', 'features_half_chain_size')
        self.n_graph_features = 0
        if dir_features_half_chain_size >= 0:
            self.n_graph_features += 2 * (2 * dir_features_half_chain_size)
        if features_half_chain_size >= 0:
            self.n_graph_features += self.n_obj_cue_channels * (
                2 * features_half_chain_size + 1)

    def normalize_dir_features(self):
        dir_features_half_chain_size = self.config.getint(
            'Graph Module', 'dir_features_half_chain_size')
        if dir_features_half_chain_size < 0:
            return
        start_idx = 0
        end_idx = dir_features_half_chain_size
        min_v = torch.min(self.all_frame_features[:, :, start_idx:end_idx])
        max_v = torch.max(self.all_frame_features[:, :, start_idx:end_idx])
        self.all_frame_features[:, :, start_idx:end_idx] = (
            self.all_frame_features[:, :, start_idx:end_idx] -
            min_v) / (max_v - min_v)

        start_idx = dir_features_half_chain_size
        end_idx = 2 * dir_features_half_chain_size
        min_v = torch.min(self.all_frame_features[:, :, start_idx:end_idx])
        max_v = torch.max(self.all_frame_features[:, :, start_idx:end_idx])
        self.all_frame_features[:, :, start_idx:end_idx] = (
            self.all_frame_features[:, :, start_idx:end_idx] -
            min_v) / (max_v - min_v)

        start_idx = 2 * dir_features_half_chain_size
        end_idx = 3 * dir_features_half_chain_size
        min_v = torch.min(self.all_frame_features[:, :, start_idx:end_idx])
        max_v = torch.max(self.all_frame_features[:, :, start_idx:end_idx])
        self.all_frame_features[:, :, start_idx:end_idx] = (
            self.all_frame_features[:, :, start_idx:end_idx] -
            min_v) / (max_v - min_v)

        start_idx = 3 * dir_features_half_chain_size
        end_idx = 4 * dir_features_half_chain_size
        min_v = torch.min(self.all_frame_features[:, :, start_idx:end_idx])
        max_v = torch.max(self.all_frame_features[:, :, start_idx:end_idx])
        self.all_frame_features[:, :, start_idx:end_idx] = (
            self.all_frame_features[:, :, start_idx:end_idx] -
            min_v) / (max_v - min_v)

    def normalize_features(self):
        dir_features_half_chain_size = self.config.getint(
            'Graph Module', 'dir_features_half_chain_size')
        features_half_chain_size = self.config.getint(
            'Graph Module', 'features_half_chain_size')
        if features_half_chain_size < 0:
            return
        start_idx = 0
        if dir_features_half_chain_size >= 0:
            start_idx += 4 * dir_features_half_chain_size
        end_idx = start_idx + self.n_obj_cue_channels * (
            2 * features_half_chain_size + 1)

        min_v = torch.min(self.all_frame_features[:, :, start_idx:end_idx])
        max_v = torch.max(self.all_frame_features[:, :, start_idx:end_idx])

        self.all_frame_features[:, :, start_idx:end_idx] = (
            self.all_frame_features[:, :, start_idx:end_idx] -
            min_v) / (max_v - min_v)


def propagation_step_undirected(in_soft_segs, n_frames, space_time_graph,
                                working_h, working_w):
    soft_segs = []
    for idx in range(n_frames):
        soft_seg = torch.zeros(working_h * working_w, 1,
                               dtype=torch.float32).cuda()
        soft_segs.append(soft_seg)

    for pivot_idx in range(n_frames):

        pivot_soft_seg = in_soft_segs[pivot_idx].clone()
        self_cues = space_time_graph.all_edge_weights[pivot_idx][
            pivot_idx].cuda()

        soft_segs[pivot_idx].add_(pivot_soft_seg * self_cues)

        indexes1 = np.arange(pivot_idx, min(pivot_idx + 15, n_frames - 1), 1)
        #indexes1 = np.arange(pivot_idx, n_frames-1, 1)
        offsets1 = np.ones(indexes1.shape, dtype=np.int32)
        indexes2 = np.arange(pivot_idx - 1, max(-1, pivot_idx - 16), -1)
        #indexes2 = np.arange(pivot_idx-1, -1, -1)
        offsets2 = np.zeros(indexes2.shape, dtype=np.int32)
        indexes = np.concatenate((indexes1, indexes2))
        offsets = np.concatenate((offsets1, offsets2))

        for idx, offset in zip(indexes, offsets):
            edge_weights = space_time_graph.all_edge_weights[pivot_idx][
                idx + offset].cuda()

            pos_edge_weights = edge_weights

            coords = space_time_graph.all_coords[pivot_idx][idx +
                                                            offset].cuda()

            dst_soft_seg = in_soft_segs[idx + offset]

            rec_soft_seg, rec_dst_soft_seg = graph_module.utils.graph_perform_projection_undirected(
                pivot_soft_seg, dst_soft_seg, pos_edge_weights, coords,
                working_h, working_w)

            rec_soft_seg[0] = 0
            rec_dst_soft_seg[0] = 0

            soft_segs[idx + offset].add_(rec_soft_seg)
            soft_segs[pivot_idx].add_(rec_dst_soft_seg)

    return soft_segs


def regression_step(in_soft_segs, n_frames, all_frame_features, working_h,
                    working_w, lambda_):
    soft_segs = []
    for pivot_idx in range(n_frames):
        # load features
        frame_features = all_frame_features[pivot_idx]
        frame_features = frame_features.cuda()

        bias_line = torch.ones(working_h * working_w, 1, dtype=torch.float32)
        bias_line = bias_line.cuda()
        frame_features = torch.cat((frame_features, bias_line), 1)
        frame_features_t = torch.t(frame_features)
        n_features = frame_features.shape[1]

        # labels
        soft_seg = in_soft_segs[pivot_idx]
        labels = soft_seg.clone()
        min_val = torch.min(labels)
        max_val = torch.max(labels)
        labels = (labels - min_val) / (max_val - min_val)

        identity = torch.eye(n_features, dtype=torch.float32)
        identity = identity.cuda()

        step1 = torch.mm(frame_features_t, frame_features) + lambda_ * identity
        step1 = step1.cpu()
        step2 = torch.inverse(step1)
        step2 = step2.cuda()
        w = torch.mm(torch.mm(step2, frame_features_t), labels)

        result = torch.mm(frame_features, w)

        result[result < 0] = 0
        min_val = torch.min(result)
        max_val = torch.max(result)
        result = (result - min_val) / (max_val - min_val)

        soft_segs.append(result)

    return soft_segs


def normalize_img_level(frames):
    n_frames = len(frames)
    for frame_idx in range(n_frames):
        frame = frames[frame_idx]
        min_value = torch.min(frame)
        max_value = torch.max(frame)
        frame = (frame - min_value) / (max_value - min_value)
        frames[frame_idx] = frame
    return frames


def get_to_save_soft_segs(soft_segs, working_h, working_w):
    n_frames = len(soft_segs)
    to_print_soft_segs = []
    for frame_idx in range(n_frames):
        to_print_soft_seg = torch.zeros(working_h,
                                        working_w,
                                        1,
                                        dtype=torch.float32)
        to_print_soft_seg[:, :, 0] = torch.reshape(soft_segs[frame_idx].cpu(),
                                                   (working_h, working_w))
        to_print_soft_segs.append(to_print_soft_seg)
    return to_print_soft_segs


def main(
    config, \
    frames, \
    seed_soft_segs,\
    object_cues,\
    fwd_flows, bwd_flows,\
    orig_height, orig_width):

    n_frames = len(frames)

    space_time_graph = SpaceTimeGraph(config, frames[:], fwd_flows, bwd_flows,
                                      object_cues)

    in_soft_segs = []
    working_w = config.getint('General', 'working_w')
    working_h = config.getint('General', 'working_h')
    for soft_seg in seed_soft_segs:
        in_soft_seg = torch.reshape(soft_seg, (working_h * working_w, 1))
        in_soft_seg = in_soft_seg.cuda()
        in_soft_segs.append(in_soft_seg)

    n_iterations = config.getint('Graph Module', 'n_iterations')
    to_save_iterations_str = config.get('Graph Module',
                                        'to_save_iterations').split(',')
    to_save_iterations = np.array([int(i) for i in to_save_iterations_str])
 
    for iteration_idx in range(n_iterations):
        print('Iteration %d of %d'%(iteration_idx, n_iterations))
        sys.stdout.flush()

        soft_segs = propagation_step_undirected(in_soft_segs, n_frames,
                                                space_time_graph, working_h,
                                                working_w)

        soft_segs_img_level = normalize_img_level(soft_segs[:])

        lambda_ = config.getfloat('Graph Module', 'lambda')
        soft_segs = regression_step(soft_segs_img_level[:],
                                    space_time_graph.n_frames,
                                    space_time_graph.all_frame_features,
                                    working_h, working_w, lambda_)

        if np.sum(to_save_iterations == iteration_idx) > 0:
            out_path = os.path.join(config.get('PATHS', 'OUT_PATH'),
                                    'iter_%d' % iteration_idx)
            to_print_soft_segs = get_to_save_soft_segs(soft_segs, working_h,
                                                       working_w)
            ike_utils.save_data.save_soft_segs_working_size(
                out_path, to_print_soft_segs)

            # save images 
            out_path = os.path.join(config.get('PATHS', 'OUT_PATH'),
                                'iter_%d_images'%iteration_idx)
            ike_utils.save_data.save_soft_segs_images(out_path, to_print_soft_segs, orig_height, orig_width)


        in_soft_segs = soft_segs

    return 
