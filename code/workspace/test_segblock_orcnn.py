import sys
sys.path.append('/esat/drejo/r0773655/thesis/code/segblocks-main/segblocks/')
sys.path.append('/esat/drejo/r0773655/thesis/code/')
sys.path.append('/esat/drejo/r0773655/thesis/code/OBBDetection/')
sys.path.append('/esat/drejo/r0773655/thesis/code/OBBDetection/mmdet/')

from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    show_result_pyplot(model, args.img, result, score_thr=args.score_thr)


if __name__ == '__main__':
    main()

'''
usage:
python ./code/workspace/test_segblock_orcnn.py \
./code/OBBDetection/demo/image_demo.jpg \
./code/OBBDetection/configs/obb/oriented_rcnn/faster_rcnn_orpn_r50_fpn_1x_ms_rr_dota10.py \
./code/OBBDetection/checkpoints/faster_rcnn_orpn_r50_fpn_1x_mssplit_rr_dota10_epoch12.pth
'''