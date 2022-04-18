# Thesis

## 0. Introduction

`PLACEHOLDER`

## 1. Structure

- `./note/`: structured note, preferably opened by [Obsidian](obsidian.md).
- `./code/`: codebase.
  - `./code/dataset_statistics/`: code about dataset evaluation, including `PLACEHOLDER`.
  - `./code/OBBDetection/`: forked from [OBBDetection](https://github.com/jbwang1997/OBBDetection).
- `./dataset/`: Download instructions can be found in [DOTA 2 dataset](https://captain-whu.github.io/DOTA/tasks.html). The folder structure is strictly organized as the original downloaded dataset.

```txt
dataset
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