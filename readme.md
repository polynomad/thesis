# Thesis

## 0. Introduction

`PLACEHOLDER`

## 1. Structure

- `./note/`: structured note, preferably opened by [Obsidian](obsidian.md).
- `./code/`: codebase.
  - `./code/dataset_statistics/`: code about dataset evaluation, including `PLACEHOLDER`.
  - `./code/OBBDetection/`: forked from [OBBDetection](https://github.com/jbwang1997/OBBDetection).
- `./data/`: Download instructions can be found in [DOTA 2 dataset](https://captain-whu.github.io/DOTA/tasks.html). The folder structure is strictly organized as the original downloaded dataset.

```txt
data
└── DOTA2_0
    ├── test-challenge
    │   ├── images
    │   │   ├── test-challenge-part1
    │   │   ├── test-challenge-part2
    │   │   ├── test-challenge-part3
    │   │   ├── test-challenge-part4
    │   │   ├── test-challenge-part5
    │   │   ├── test-challenge-part6
    │   │   ├── test-challenge-part7
    │   │   └── test-challenge-part8
    │   └── meta
    ├── test-dev
    │   ├── images
    │   │   ├── part10
    │   │   ├── part3
    │   │   ├── part4
    │   │   ├── part5
    │   │   ├── part6
    │   │   ├── part7
    │   │   ├── part8
    │   │   └── part9
    │   └── meta
    ├── train
    │   ├── images
    │   │   ├── part4
    │   │   ├── part5
    │   │   └── part6
    │   ├── labelTxt-v2.0
    │   │   ├── DOTA-v2.0_train
    │   │   └── DOTA-v2.0_train_hbb
    │   └── meta
    └── val
        ├── images
        │   └── part2
        ├── labelTxt-v2.0
        │   ├── DOTA-v2.0_val
        │   └── DOTA-v2.0_val_hbb
        └── meta
```

## 2. Usage

### 2.0 Prequisites

### 2.1 Dataset Evaluation

```bash
python3 ./code/dataset_statistics/evaluate_dataset.py --dataset_dir ./dataset/
```

### 2.2 Train

First, DOTA dataset is to be splitted. `PLACEHOLDER`. The following code is an example. For further information, please refer to [BboxToolkit documentation](./code/OBBDetection/BboxToolkit/README.md) and [BboxToolkit usage](./code/OBBDetection/BboxToolkit/USAGE.md).

```bash
python ./code/OBBDetection/BboxToolkit/tools/img_split.py \
--load_type dota \
--base_json ./code/OBBDetection/BboxToolkit/tools/split_configs/dota2_0/ss_train.json \
--img_dirs ./data/DOTA2_0/train/images/part4 \
--ann_dirs ./data/DOTA2_0/train/labelTxt-v2.0/DOTA-v2.0_train \
--save_dir ./data/DOTA2_0_split_ss/train/images/part4
```

### 2.3 Val

```bash
python ./code/OBBDetection/demo/image_demo.py \
./code/OBBDetection/demo/dota_demo.jpg \
./code/OBBDetection/configs/obb/oriented_rcnn/faster_rcnn_orpn_r50_fpn_1x_ms_rr_dota10.py \
./code/OBBDetection/checkpoints/faster_rcnn_orpn_r50_fpn_1x_mssplit_rr_dota10_epoch12.pth
```