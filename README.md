# Iterative Knowledge Exchange (IKE) 

This repository contains the code associated to paper [Iterative Knowledge Exchange Between Deep Learning and Space-Time Spectral Clustering for Unsupervised Segmentation in Videos](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9573434) by Emanuela Haller, Adina Magda Florea and Marius Leordeanu. 

The paper was published in Transactions on Pattern Analysis and Machine Intelligence (TPAMI), Vol. 44, No. 11, November 2022. 

## Method Overview 

![IKE overview](/images/ike.png)

We propose a dual Iterative Knowledge Exchange (IKE) model coupling space-time spectral clustering with deep object segmentation, able to learn without any human annotation. The graph module exploits the spatio-temporal consistencies inherent in a video sequence but has no access to deep features. The deep segmentation module exploits high-level image representations but requires a supervisory signal. The two components are complementary and together form an efficient self-supervised system able to discover and learn to segment the primary object of a video sequence. The space-time graph is the first to play the teacher role by discovering objects unsupervised. Next, the deep model is trained from scratch for each video sequence, using the graph's output as pseudo-ground truth. At the second cycle, the graph module incorporates knowledge from the segmentation model, adding the powerful deep learned features that correspond to each of its nodes. The process in which the graph and the deep net exchange knowledge in this manner repeats over several cycles until convergence is reached. Below, we present the architecture of the proposed system, highlighting its two main modules: Graph Module and Network Module.

## Extract Optical Flow 

We have considered the RAFT optical flow solution from https://github.com/princeton-vl/RAFT.
Clone the repository in *RAFT* folder, download the pretrained models (we have considered the one trained on FlyingThings3D) and run the following command (replacing the paths):

```
python extract_optical_flow.py --frames_path=/root/frames_path --out_path=/root/of_path --models=./RAFT/models/raft-things.pth
```

Will generate both forward and backward optical flow data for video stored at *frames_path* (raw video frames, '%05d.jpg'%(idx)).
The forward optical flow will be stored in *out_path/fwd_of* (frame idx => frame idx+1 in *'%05d_%05d.flo'%(idx, idx+1))*, and backward flow will be stored in *out_path/bwd_of* (frame idx+1 => frame idx in *'%05d_%05d.flo'%(idx, idx+1)*).

## Run Graph Module 

Run the following command:
```
python run_graph_module.py --frames_path=/root/frames_path --of_path=/root/of_path --out_path=/root/out_path --config=./config.ini
```

Will generate the results of our iterative Graph Module, storing iteration results in pt files required for next cycles (in *out_path/iter_idx*, where idx is the iteration index), but also in png format (in *out_path/iter_idx_images*, where idx is the iteration index).

If running the Graph Module for the second cycle, you will need to provide the cues of the Network Module in *ike_features_paths* from *config.ini*.

Check *config.ini* for other configuration parameters.

## Run Network Module
```
python main.py --frames_path=/root/frames_path --pseudo_gt_path=/root/pseudo_gt --out_path=/root/out_path --config=./config.ini
```

Will train the deep model using the provided pseudo labels. Checkpoints will be stored in *out_path/models*, while resulted predictions in *out_path/results_pt* and *out_path/results_images*.

Check *config.ini* for other configuration parameters. 

## If you intend to use our work please cite this project as:
```
@article{haller2021iterative,
  title={Iterative Knowledge Exchange Between Deep Learning and Space-Time Spectral Clustering for Unsupervised Segmentation in Videos},
  author={Haller, Emanuela and Florea, Adina Magda and Leordeanu, Marius},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={44},
  number={11},
  pages={7638--7656},
  year={2021},
  publisher={IEEE}
}
```





