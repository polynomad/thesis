# Thesis

## 0. Introduction

`PLACEHOLDER`

## 1. Structure

If necessary, detailed introduction is provided in `readme.md` of different folders. 

- `./note/`: structured note, preferably opened by [Obsidian](obsidian.md).
- `./code/`: codebase.
  - `./code/dataset_statistics/`: code about dataset evaluation, including `PLACEHOLDER`.
  - `./code/OBBDetection/`: forked from [OBBDetection](https://github.com/jbwang1997/OBBDetection).
- `./data/`: Download instructions can be found in [DOTA 2 dataset](https://captain-whu.github.io/DOTA/tasks.html). The folder structure is strictly organized as the original downloaded dataset.

```txt
data
└── DOTA2_0
    ├── test-challenge
    │   ├── images                   ## 6053 images
    │   │   ├── test-challenge-part1 #520 images
    │   │   ├── test-challenge-part2 #576 images
    │   │   ├── test-challenge-part3 #624 images
    │   │   ├── test-challenge-part4 #840 images
    │   │   ├── test-challenge-part5 #920 images
    │   │   ├── test-challenge-part6 #1016 images
    │   │   ├── test-challenge-part7 #1000 images
    │   │   └── test-challenge-part8 #557 images
    │   └── meta
    ├── test-dev
    │   ├── images                   ##1855 images
    │   │   ├── part10               #179 images
    │   │   ├── part3                #444 images
    │   │   ├── part4                #252 images
    │   │   ├── part5                #264 images
    │   │   ├── part6                #144 images
    │   │   ├── part7                #132 images
    │   │   ├── part8                #175 images
    │   │   └── part9                #265 images
    │   └── meta
    ├── train
    │   ├── images                   ##419 images
    │   │   ├── part4                #144 images
    │   │   ├── part5                #210 images
    │   │   └── part6                #65  images
    │   ├── labelTxt-v2.0
    │   │   ├── DOTA-v2.0_train      #1830 annotations
    │   │   └── DOTA-v2.0_train_hbb  #1830 annotations
    │   └── meta
    └── val
        ├── images                   ##135 images
        │   └── part2                #135 images
        ├── labelTxt-v2.0
        │   ├── DOTA-v2.0_val        #593 annotations
        │   └── DOTA-v2.0_val_hbb    #593 annotations
        └── meta
```

## 2. Usage

### 2.0 Prequisites


#### 2.0.1 Installations of env
Install the necessary dependencies via conda venv. 

```bash
# after installation of anaconda
conda create --name obbdetection
source activate obbdetection
# <ESAT PSI server specific start>
conda install gcc_linux-64 # for solving not compatible compiler issue, in ESAT
conda install gxx_linux-64 # for solving not compatible compiler issue, in ESAT
# <ESAT PSI server specific end>
cd code/OBBDetection
pip install -v -e .
cd BboxToolkit
pip install -v -e . 
cd ../../..
```

Then, set `~/.bashrc` to auto activate the venv when logging in (just add `source activate obbdetection` in the last line of `~/.bashrc`).

#### 2.0.2 Note Compilation

### 2.1 Dataset Evaluation

#### 2.1.1 evaluating the bounding box coverage of the entire dataset

```bash
python ./code/OBBDetection/BboxToolkit/tools/bbox_coverage.py \
--load_type dota \
--classes dota2.0 \
--img_dirs ./data/DOTA2_0/train/images/part4 \
--ann_dirs ./data/DOTA2_0/train/labelTxt-v2.0/DOTA-v2.0_train
```

#### 2.1.2 evaluating the upper limit of recognized objects with window rejection

```bash
python ./code/OBBDetection/BboxToolkit/tools/window_rej.py \
--load_type dota \
--base_json ./code/OBBDetection/BboxToolkit/tools/split_configs/dota2_0/ss_train.json \
--img_dirs ./data/DOTA2_0/train/images/part4 \
--ann_dirs ./data/DOTA2_0/train/labelTxt-v2.0/DOTA-v2.0_train \
--save_dir ./data/DOTA2_0_split_default/train/part4 \
--obj_rej_thres 0
```

#### 2.1.3 splitting images into sub images (contained in the original codebase)

Before training (a sub-image based aerial image detector), DOTA dataset is to be splitted. `PLACEHOLDER`. The following code is an example. For further information, please refer to [BboxToolkit documentation](./code/OBBDetection/BboxToolkit/README.md) and [BboxToolkit usage](./code/OBBDetection/BboxToolkit/USAGE.md).

```bash
python ./code/OBBDetection/BboxToolkit/tools/img_split.py \
--load_type dota \
--base_json ./code/OBBDetection/BboxToolkit/tools/split_configs/dota2_0/ss_train.json \
--img_dirs ./data/DOTA2_0/train/images/part4 \
--ann_dirs ./data/DOTA2_0/train/labelTxt-v2.0/DOTA-v2.0_train \
--save_dir ./data/DOTA2_0_split_default/train/part4

python ./code/OBBDetection/BboxToolkit/tools/img_split.py \
--load_type dota \
--base_json ./code/OBBDetection/BboxToolkit/tools/split_configs/dota2_0/ss_train.json \
--img_dirs ./data/DOTA2_0/train/images/part5 \
--ann_dirs ./data/DOTA2_0/train/labelTxt-v2.0/DOTA-v2.0_train \
--save_dir ./data/DOTA2_0_split_default/train/part5

python ./code/OBBDetection/BboxToolkit/tools/img_split.py \
--load_type dota \
--base_json ./code/OBBDetection/BboxToolkit/tools/split_configs/dota2_0/ss_train.json \
--img_dirs ./data/DOTA2_0/train/images/part6 \
--ann_dirs ./data/DOTA2_0/train/labelTxt-v2.0/DOTA-v2.0_train \
--save_dir ./data/DOTA2_0_split_default/train/part6
```

To visualize: 

```bash
python ./code/OBBDetection/BboxToolkit/tools/visualize.py \
--load_type dota \
--base_json ./code/OBBDetection/BboxToolkit/tools/split_configs/dota2_0/ss_train.json \
--img_dirs ./data/DOTA2_0/train/images/part4 \
--ann_dirs ./data/DOTA2_0/train/labelTxt-v2.0/DOTA-v2.0_train \
--save_dir ./data/DOTA2_0_split_ss/train/images/part4
```

### 2.2 Train

### 2.3 Val

```bash
python ./code/OBBDetection/demo/image_demo.py \
./code/OBBDetection/demo/dota_demo.jpg \
./code/OBBDetection/configs/obb/oriented_rcnn/faster_rcnn_orpn_r50_fpn_1x_ms_rr_dota10.py \
./code/OBBDetection/checkpoints/faster_rcnn_orpn_r50_fpn_1x_mssplit_rr_dota10_epoch12.pth
```