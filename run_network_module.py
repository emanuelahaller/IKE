import os 
import argparse
import configparser

import network_module.main

def preprocess_config_file(config, args):

    config.set('PATHS', 'FRAMES_PATH', args.frames_path)
    config.set('PATHS', 'PSEUDO_GT_PATH', args.pseudo_gt_path)
    config.set('PATHS', 'OUT_PATH', os.path.join(args.out_path, 'predictions'))
    config.set('PATHS', 'OUT_PATH_CHECKPOINTS', os.path.join(args.out_path, 'checkpoints'))

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_path', help="video frames")
    parser.add_argument('--pseudo_gt_path', help="pseudo gt segmentation masks")
    parser.add_argument('--out_path', help="results path")
    parser.add_argument('--config_file', help="config file path")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_file)
    preprocess_config_file(config, args)
    
    network_module.main.run(config)


