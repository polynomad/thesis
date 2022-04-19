def add_parser(parser):
    #argument for loading data
    parser.add_argument('--log_dirs', type=str, default=None,
                        help='directory of log file after the splitting of image')
    parser.add_argument('--ann_dirs', nargs='+', type=str, default=None,
                        help='annotations dirs')
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
    #argument for saving data
    parser.add_argument('--save_dir', type=str, default=None,
    					help='directory to save evaluation results')


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

    if args.base_json is not None:
        with open(args.base_json, 'r') as f:
            prior_config = json.load(f)

        for action in parser._actions:
            if action.dest not in prior_config or \
               not hasattr(action, 'default'):
                continue
            action.default = prior_config[action.dest]
            args = parser.parse_args()

    # assert arguments
    assert args.load_type is not None, "argument load_type can't be None"
    assert args.img_dirs is not None, "argument img_dirs can't be None"
    args.img_dirs = abspath(args.img_dirs)
    assert args.ann_dirs is None or len(args.ann_dirs) == len(args.img_dirs)
    args.ann_dirs = abspath(args.ann_dirs)
    if args.classes is not None and osp.isfile(args.classes):
        args.classes = abspath(args.classes)
    assert args.prior_annfile is None or args.prior_annfile.endswith('.pkl')
    args.prior_annfile = abspath(args.prior_annfile)
    assert args.merge_type in ['addition', 'replace']
    assert len(args.sizes) == len(args.gaps)
    assert len(args.sizes) == 1 or len(args.rates) == 1
    assert args.save_ext in bt.img_exts
    assert args.iof_thr >= 0 and args.iof_thr < 1
    assert args.iof_thr >= 0 and args.iof_thr <= 1
    assert not osp.exists(args.save_dir), \
            f'{osp.join(args.save_dir)} already exists'
    args.save_dir = abspath(args.save_dir)
    return args

def main(image_width, image_height, bbox):
	"""calculate bounding box coverage of one single image.

	:image_width: image width, in pixels, integer
	:image_height: image height, in pixels, integer
	:bbox: 

	"""

if __name__ == '__main__':
	main()