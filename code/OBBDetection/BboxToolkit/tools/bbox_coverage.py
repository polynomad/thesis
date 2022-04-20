import argparse
import json
import os.path as osp
import BboxToolkit as bt


def add_parser(parser):
    #argument for loading data
    parser.add_argument('--load_type', type=str, default=None,
                        help='loading function type')
    parser.add_argument('--classes', type=str, default=None,
                        help='the classes for loading data')
    parser.add_argument('--ann_dirs', nargs='+', type=str, default=None,
                        help='annotations dirs')
    parser.add_argument('--img_dirs', nargs='+', type=str, default=None,
                        help='images dirs, must give a value')
    #argument for splitting image
    parser.add_argument('--sizes', nargs='+', type=int, default=[1024],
                        help='the sizes of sliding windows')
    parser.add_argument('--gaps', nargs='+', type=int, default=[512],
                        help='the steps of sliding widnows')
    parser.add_argument('--rates', nargs='+', type=float, default=[1.],
                        help='same as DOTA devkit rate, but only change windows size')
    parser.add_argument('--img_rate_thr', type=float, default=0.6,
                        help='the minimal rate of image in window and window')
    parser.add_argument('--iof_thr', type=float, default=0.7,
                        help='the minimal iof between a object and a window')
    parser.add_argument('--no_padding', action='store_true',
                        help='not padding patches to regular size')
    parser.add_argument('--padding_value', nargs='+',type=int, default=[0],
                        help='padding value, 1 or channel number')


def abspath(path):
    if isinstance(path, (list, tuple)):
        return type(path)([abspath(p) for p in path])
    if path is None:
        return path
    if isinstance(path, str):
        return osp.abspath(path)
    raise TypeError('Invalid path type.')


def parse_args():
    parser = argparse.ArgumentParser(description='evaluate bounding box coverage')
    add_parser(parser)
    args = parser.parse_args()

    # assert arguments
    assert args.load_type is not None, "argument load_type can't be None"
    assert args.img_dirs is not None, "argument img_dirs can't be None"
    args.img_dirs = abspath(args.img_dirs)
    assert args.ann_dirs is None or len(args.ann_dirs) == len(args.img_dirs)
    args.ann_dirs = abspath(args.ann_dirs)
    if args.classes is not None and osp.isfile(args.classes):
        args.classes = abspath(args.classes)
    assert len(args.sizes) == len(args.gaps)
    assert len(args.sizes) == 1 or len(args.rates) == 1
    assert args.iof_thr >= 0 and args.iof_thr < 1
    assert args.iof_thr >= 0 and args.iof_thr <= 1
    return args

def main():
    args = parse_args()
    infos, img_dirs = [], []
    load_func = getattr(bt.datasets, 'load_'+args.load_type)
    for img_dir, ann_dir in zip(args.img_dirs, args.ann_dirs):
        _infos, classes = load_func(
            img_dir=img_dir,
            ann_dir=ann_dir,
            classes=args.classes)
        _img_dirs = [img_dir for _ in range(len(_infos))]
        infos.extend(_infos)
        img_dirs.extend(_img_dirs)
    print('hello')

if __name__ == '__main__':
    main()