import sys 
import os 
import numpy as np
import torch
import graph_module.initialization
import graph_module.space_time_graph
import ike_utils.get_data
import ike_utils.save_data
import ike_utils.flow

def get_list_of_features(config):
    """
        [out] features_path - list of folders containing features for all dataset videos 
    """
    features_paths = []
    # get IKE features
    if config.has_option('Graph Module', 'ike_features_paths'):
        ike_features_paths = config.get('Graph Module', 'ike_features_paths')
        if len(ike_features_paths) == 0:
            ike_features_paths = []
        else:
            ike_features_paths = ike_features_paths.split('\n')
            features_paths.extend(ike_features_paths)
    return features_paths

def load_video_features(features_path, n_frames):

    video_features = []
    if len(features_path) == 0:
        return video_features
    for frame_idx in range(n_frames):
        frame_features = []

        for feature_path in features_path:
            frame_feature_path = os.path.join(feature_path, '%05d.pt' % frame_idx)
            frame_feature = torch.load(frame_feature_path)
            frame_features.append(frame_feature)
        frame_features = torch.cat(frame_features, dim=2)

        video_features.append(frame_features)
    return video_features


def run_video(config, frames, orig_height, orig_width):
    n_frames = len(frames)
    n_iterations = config.getint('Graph Module', 'n_iterations')

    to_save_iterations_str = config.get('Graph Module',
                                        'to_save_iterations')
    if len(to_save_iterations_str) == 0:
        to_save_iterations = np.arange(-1, n_iterations)
    else:
        to_save_iterations_str = to_save_iterations_str.split(',')
        to_save_iterations = np.array([int(i) for i in to_save_iterations_str])
   
    height = config.getint('General', 'working_h')
    width = config.getint('General', 'working_w')

    # generate seeds
    seed_soft_segs = graph_module.initialization.get_initialization(
        config, n_frames)

    # save seeds
    if np.sum(to_save_iterations == -1) > 0:
        # save tensors 
        out_path = os.path.join(config.get('PATHS', 'OUT_PATH'),
                                'iter_-1')
        ike_utils.save_data.save_soft_segs_working_size(out_path, seed_soft_segs)
        # save images 
        out_path = os.path.join(config.get('PATHS', 'OUT_PATH'),
                                'iter_-1_images')
        ike_utils.save_data.save_soft_segs_images(out_path, seed_soft_segs, orig_height, orig_width)

    # get initial features
    features_path = get_list_of_features(config)
    video_features = load_video_features(features_path, n_frames)

    # get optical flow
    fwd_flows = ike_utils.flow.get_video_optical_flow(config.get('PATHS','FWD_OF_PATH'), n_frames, height, width)
    bwd_flows = ike_utils.flow.get_video_optical_flow(config.get('PATHS','BWD_OF_PATH'), n_frames, height, width)

    # run graph module iterations 
    graph_module.space_time_graph.main(config, frames, seed_soft_segs, video_features, fwd_flows, bwd_flows,
        orig_height, orig_width)
        
def run(config):

    main_out_path = config.get('PATHS', 'OUT_PATH')
    os.makedirs(main_out_path, exist_ok=True)

    frames_path = config.get('PATHS', 'FRAMES_PATH')

    print('Run IKE - Graph Module on %s' % (frames_path))
    sys.stdout.flush()

    height, width = ike_utils.get_data.get_video_resolution(frames_path)
    frames = ike_utils.get_data.get_video_frames_bgr(config, frames_path)

    run_video(config, frames, height, width)