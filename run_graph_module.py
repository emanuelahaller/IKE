import os 
import argparse
import configparser

import graph_module.main

def get_n_features(config):
    n_features = 0
    if config.has_option('Graph Module', 'ike_features_paths'):
        ike_features_paths = config.get('Graph Module', 'ike_features_paths')
        if len(ike_features_paths) > 0:
            ike_features_paths = ike_features_paths.split('\n')
            n_features += len(ike_features_paths)
    return n_features

def preprocess_config_file(config, args):
    n_features = get_n_features(config)
    if n_features == 0:
        config.set('Graph Module', 'features_half_chain_size', '-1')

    config.set('PATHS', 'FRAMES_PATH', args.frames_path)
    config.set('PATHS', 'FWD_OF_PATH', os.path.join(args.of_path, 'fwd_of'))
    config.set('PATHS', 'BWD_OF_PATH', os.path.join(args.of_path, 'bwd_of'))
    config.set('PATHS', 'OUT_PATH', args.out_path)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_path', help="video frames")
    parser.add_argument('--of_path', help="optical flow data")
    parser.add_argument('--out_path', help="results path")
    parser.add_argument('--config_file', help="config file path")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_file)
    preprocess_config_file(config, args)

    graph_module.main.run(config)
