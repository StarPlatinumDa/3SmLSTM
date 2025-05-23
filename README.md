# 3SmLSTM
This is the official PyTorch implementation of our paper 3SmLSTM: Symmetric Feature Mixing MatrixLSTM with Specific Skeletal Connectivity Encoding for Skeleton-Based Action Recognition
The paper will be published in ICIC 2025.

## Efficiency
| Model | Parameters | FLOPs | Acc on NTU 120 X-Sub | Acc on NTU 120 X-Set|
| -------- | ------- | -------- | -------- | -------- |
| 3SmLSTM | 2.0M | 2.60G |  89.9 | 91.2 |

####  3SmLSTM architecture
<p align="center">
   <img src="full.png" alt="drawing" width="800"/>
</p>


#  Preparation

### Download datasets.

#### There are 3 datasets to download:

- NTU RGB+D 60 Skeleton
- NTU RGB+D 120 Skeleton
- NW-UCLA

#### NTU RGB+D 60 and 120

1. Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:
   1. `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)
   2. `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
   3. Extract above files to `./data/nturgbd_raw`

#### NW-UCLA

1. Download dataset from [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN)
2. Move `all_sqe` to `./data/NW-UCLA`

### Data Processing

#### Directory Structure

Put downloaded data into the following directory structure:

```
- data/
  - NW-UCLA/
    - all_sqe
      ... # raw data of NW-UCLA
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
```

#### Generating Data

- Generate NTU RGB+D 60 or NTU RGB+D 120 dataset:

```
 cd ./data/ntu # or cd ./data/ntu120
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```

# Evaluation

We provide the [pretrained model weights](https://github.com/StarPlatinumDa/3SmLSTM/tree/main/pretrained%20weights) for NTURGB+D 60 and NTURGB+D 120 benchmarks.

To use the pretrained weights for evaluation, please run the following command:

```
python main.py --weights pretrained weights/ntu 60/joint/runs-137-40778.pt --phase test --save-score True --config config/nturgbd-cross-subject/joint.yaml --device 0 --start-epoch 137
```

# Training

```
python main.py --config config/nturgbd-cross-subject/joint.yaml --device 0 --base-lr 2.5e-2
```

## Acknowledgements

This repo is based on [Hyperformer](https://github.com/ZhouYuxuanYX/Hyperformer) and [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN). The training strategy is based on CTR-GCN.

Thanks to the original authors for their great work!

## Citation

Please cite this work if you find it useful.




