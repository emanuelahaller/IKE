import os 
import sys 
import cv2
import torch 
import numpy as np
from torch.utils.data import Dataset, DataLoader

import ike_utils.get_data
import ike_utils.save_data

import network_module.model

class SegmentationDataset(Dataset):
    def __init__(self, in_frames=[], in_segmentations=[], bm_th=0.5, sat_min_th=0, sat_max_th=1):
        super(SegmentationDataset, self).__init__()

        self.frames = []
        self.annotations = []
        for idx in range(len(in_frames)):
            self.frames.append(in_frames[idx].numpy())
            self.annotations.append(in_segmentations[idx].numpy())

        self.bm_th = bm_th
        self.sat_min_th = sat_min_th
        self.sat_max_th = sat_max_th

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        
        frame = np.copy(self.frames[index])
        annotation = np.copy(self.annotations[index])
        annotation = annotation[:,:,0]
        
        soft_annotation_ = np.copy(annotation)
        soft_annotation_[soft_annotation_<=self.sat_min_th] = 0
        soft_annotation_[soft_annotation_>=self.sat_max_th] = 1

        annotation[annotation>=self.bm_th] = 1
        annotation[annotation<self.bm_th] = 0

        soft_annotation = torch.zeros(annotation.shape[0], annotation.shape[1], 2, dtype=torch.float32)
        soft_annotation[:,:,1] = torch.tensor(soft_annotation_)
        soft_annotation[:,:,0] = 1 - soft_annotation[:,:,1]
        soft_annotation = soft_annotation.permute(2, 0, 1)
        
        frame = torch.tensor(frame, dtype=torch.float32)
        annotation = torch.tensor(annotation, dtype=torch.float32)

        frame = frame.permute(2, 0, 1)
        '''
    
        frame = frame / 255
    
        frame[0,:,:] = frame[0,:,:] - 0.485
        frame[1,:,:] = frame[1,:,:] - 0.456
        frame[2,:,:] = frame[2,:,:] - 0.406
        frame[0,:,:] = frame[0,:,:] / 0.229
        frame[1,:,:] = frame[1,:,:] / 0.224
        frame[2,:,:] = frame[2,:,:] / 0.225
        '''
        annotation[annotation>=self.bm_th] = 1
        annotation[annotation<self.bm_th] = 0
        
        return frame, annotation, soft_annotation

class SegmentationDataset_forSave(Dataset):
    def __init__(self, in_frames=[]):
        super(SegmentationDataset_forSave, self).__init__()

        self.frames = []
      
        for idx in range(len(in_frames)):
            self.frames.append(in_frames[idx].numpy())
        
    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
       
        frame = np.copy(self.frames[index])

        frame = torch.tensor(frame, dtype=torch.float32)
        frame = frame.permute(2, 0, 1)
        '''
    
        frame = frame / 255
        frame[0,:,:] = frame[0,:,:] - 0.485
        frame[1,:,:] = frame[1,:,:] - 0.456
        frame[2,:,:] = frame[2,:,:] - 0.406
        frame[0,:,:] = frame[0,:,:] / 0.229
        frame[1,:,:] = frame[1,:,:] / 0.224
        frame[2,:,:] = frame[2,:,:] / 0.225
        '''
        return frame

def load_video_pseudo_gt(pseudo_gt_path, n_frames):
    pseudo_gts = []
    for frame_idx in range(n_frames):
        frame_path = os.path.join(pseudo_gt_path, '%05d.pt'%frame_idx)
        frame_pseudo_gt = torch.load(frame_path)      
        pseudo_gts.append(frame_pseudo_gt)
    return pseudo_gts

def diceloss(outputs, targets):
    outputs = torch.softmax(outputs, 1)
    
    s_comb = torch.sum(outputs[:,1,:,:] * targets[:,1,:,:], (1,2))
    s_outputs = torch.sum(outputs[:,1,:,:], (1,2))
    s_targets_1 = torch.sum(targets[:,1,:,:], (1,2))
    dl_loss_1 = 1 - (2*s_comb+1) / (s_outputs+s_targets_1+1)

    s_comb = torch.sum(outputs[:,0,:,:] * targets[:,0,:,:], (1,2))
    s_outputs = torch.sum(outputs[:,0,:,:], (1,2))
    s_targets_0 = torch.sum(targets[:,0,:,:], (1,2))
    dl_loss_0 = 1 - (2*s_comb+1) / (s_outputs+s_targets_0+1)
    
    dl_loss = (dl_loss_1 + dl_loss_0) * 0.5
    dl_loss = torch.mean(dl_loss)
 
    return dl_loss

def train(config, video_train_dataloader, learning_rate, n_epochs, to_save_epochs, video_to_save_dataloader):
    net = network_module.model.ResNet_UNet(in_channels=3, out_channels=2)
    net = net.cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.5, 0.5]).cuda())
    softmax = torch.nn.Softmax(dim=1)
    for epoch in range(n_epochs):
        print('Epoch %d out of %d'%(epoch+1, n_epochs))
        net.train()
        train_loss = 0 
        n_train_batches = 0 
        
        for batch_idx, batch_data in enumerate(video_train_dataloader):
            
            imgs, labels, soft_labels = batch_data
            imgs = imgs.cuda()
            labels = labels.cuda()
            soft_labels = soft_labels.cuda()

            optimizer.zero_grad()
            net_out = net(imgs)
            
            loss1 = criterion(net_out, labels.long())
            loss2 = diceloss(net_out, soft_labels)
            loss = loss1+loss2 
            
            loss.backward()
            optimizer.step() 

            n_train_batches+=1 
            train_loss = train_loss + loss.item() 

        train_loss /= n_train_batches
      
        if np.sum(to_save_epochs == (epoch+1))>0:
            # save checkpoint 
            path = os.path.join(config.get('PATHS', 'OUT_PATH_CHECKPOINTS'), 'epoch_%05d.pt'%(epoch+1))
            torch.save(net.state_dict(), path)   
            
            # save pts & images 
            pts_path = os.path.join(config.get('PATHS', 'OUT_PATH'), 'epoch_%05d'%(epoch+1))
            os.makedirs(pts_path, exist_ok=True)
            images_path = os.path.join(config.get('PATHS', 'OUT_PATH'), 'epoch_%05d_images'%(epoch+1))
            os.makedirs(images_path, exist_ok=True)

            net.eval()
            frame_idx = 0
            for batch_idx, batch_data in enumerate(video_to_save_dataloader):
               
                imgs = batch_data
                imgs = imgs.cuda()
                net_out = net(imgs)

                net_out = softmax(net_out)
        
                net_out = net_out.detach()
                net_out = net_out[0,1,:,:].unsqueeze(2)
                torch.save(net_out, os.path.join(pts_path, '%05d.pt'%frame_idx))

                img = net_out.cpu().numpy()
                cv2.imwrite(os.path.join(images_path, '%05d.png'%frame_idx), np.uint8(img*255))

                frame_idx += 1
                

            
def run_video(config, orig_height, orig_width):
    
    frames_path = config.get('PATHS', 'FRAMES_PATH')
    frames = ike_utils.get_data.get_video_frames_rgb(config, frames_path)
  
    pseudo_gt_path = config.get('PATHS', 'PSEUDO_GT_PATH')
    segmentations = load_video_pseudo_gt(pseudo_gt_path, len(frames))
   
    video_train_dataset = SegmentationDataset(in_frames=frames, in_segmentations=segmentations)
    video_train_dataloader = DataLoader(video_train_dataset, batch_size=config.getint('Network Module', 'batch_size'), shuffle=True, drop_last=True)
    video_to_save_dataset = SegmentationDataset_forSave(in_frames=frames)
    video_to_save_dataloader = DataLoader(video_to_save_dataset, batch_size=1, shuffle=False, drop_last=False)

    learning_rate = config.getfloat('Network Module', 'learning_rate')
    n_epochs = config.getint('Network Module', 'n_epochs')
   
    to_save_epochs_str = config.get('Network Module',
                                        'to_save_epochs')
    if len(to_save_epochs_str) == 0:
        to_save_epochs = np.arange(0, n_epochs)+1
    else:
        to_save_epochs_str = to_save_epochs_str.split(',')
        to_save_epochs = np.array([int(i) for i in to_save_epochs_str])

    train(config, video_train_dataloader, learning_rate, n_epochs, to_save_epochs, video_to_save_dataloader)
    
def run(config):

    main_out_path = config.get('PATHS', 'OUT_PATH')
    os.makedirs(main_out_path, exist_ok=True)

    checkpoints_main_out_path = config.get('PATHS', 'OUT_PATH_CHECKPOINTS')
    os.makedirs(checkpoints_main_out_path, exist_ok=True)
    
    frames_path = config.get('PATHS', 'FRAMES_PATH')
    pseudo_gt_path = config.get('PATHS', 'PSEUDO_GT_PATH')

    print('Run IKE - Network Module on %s -- annotations from %s' % (frames_path, pseudo_gt_path))
    sys.stdout.flush()
    height, width = ike_utils.get_data.get_video_resolution(frames_path)
    run_video(config, height, width)