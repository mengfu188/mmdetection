import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
import os
import time
import torch
from mmdet.apis import init_detector, inference_detector, show_result
from tqdm import tqdm
import mmcv


# float calcIOU(FaceInfo box1, FaceInfo box2, string mode)
# {
#     int maxX = max(box1.x[0], box2.x[0]);
#     int maxY = max(box1.y[0], box2.y[0]);
#     int minX = min(box1.x[1], box2.x[1]);
#     int minY = min(box1.y[1], box2.y[1]);
#     int width = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
#     int height = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
#     int inter = width * height;
#     if (!mode.compare("union"))
#         return float(inter) / (box1.area + box2.area - float(inter));
#     else if (!mode.compare("min"))
#         return float(inter) / (box1.area < box2.area ? box1.area : box2.area);
#     else
#         return 0;
# }


def calcIOU(box1, box2, mode):
    """box [xmin, ymin, xmax, ymax]"""
    minx = max(box1[0], box2[0])
    miny = max(box1[1], box2[1])
    maxx = min(box1[2], box2[2])
    maxy = min(box1[3], box2[3])

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    width = maxx - minx + 1 if maxx - minx + 1 > 0 else 0
    height = maxy - miny + 1 if maxy - miny + 1 > 0 else 0
    inter = width * height

    if mode == 'union':
        return inter / (area1 + area2 - inter)
    elif mode == 'min':
        return inter / (area1 if area1 < area2 else area2)
    else:
        return 0


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default='configs/oid/faster_rcnn_r50_fpn_2x_Human_head_v2.py')
    parser.add_argument('checkpoint', default='work_dirs/faster_rcnn_r50_fpn_1x_Human_head_v2/latest.pth')
    parser.add_argument('--out_dir', default='/datasets/extract_image')
    parser.add_argument('--score_thr', type=float, default=0.5)
    parser.add_argument('--min_size', type=int, default=30, help='小目标最小size')
    parser.add_argument('--expand_ratio', type=float, default=0, help='扩大倍数')
    parser.add_argument('--target_classes', default=[0], help='需要裁剪的类别')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument(
        '--camera-id', default=0, help='camera device id')
    parser.add_argument(
        '--skip-step', type=int, default=2)
    parser.add_argument('--rot90', default=0, type=int)
    parser.add_argument('--roi', type=int, nargs='+', default=[])
    parser.add_argument('--roi-threshold', type=float, default=0.5)
    parser.add_argument('--show', action='store_true')

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    camera = cv2.VideoCapture(args.camera_id)
    fps = camera.get(cv2.CAP_PROP_FPS)
    width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    camera_count = camera.get(cv2.CAP_PROP_FRAME_COUNT)

    print(f'camera fps {fps}, width {width}, height {height}, frame count {camera_count}')

    if not args.roi:
        args.roi = [0, 0, int(width-1), int(height-1)]

    frame_rate = 0
    start_time = time.time()
    frame_count = 0
    total_count = 0
    prog_bar = mmcv.ProgressBar(camera_count)

    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        grab_count = 0
        while True:
            ret_val, img = camera.read()
            grab_count += 1
            frame_count += 1
            total_count += 1
            prog_bar.update()
            if grab_count == args.skip_step:
                break

        result = inference_detector(model, img)
        result = [x if i in args.target_classes else np.zeros((0, 5)) for i, x in enumerate(result) ]
        result = [x[x[:, 4] > args.score_thr] for i, x in enumerate(result)]
        flag = False
        for obj in result:
            for bbox in obj:
                if calcIOU(bbox, args.roi, 'min') > args.roi_threshold:
                    flag=True
                    break
        if not flag:
            continue

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

        end_time = time.time()
        if (end_time - start_time) >= 1:
            frame_rate = int(frame_count / (end_time - start_time))
            start_time = time.time()
            frame_count = 0

        cv2.putText(img, str(frame_rate) + " fps", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    thickness=2, lineType=2)
        cv2.rectangle(img, (args.roi[0], args.roi[1]), (args.roi[2], args.roi[3]), (255, 0, 0), 1)

        filename = time.strftime("%d_%H:%M:%S.jpg", time.gmtime(total_count / fps))
        filepath = os.path.join(args.out_dir, filename)
        show_result(
            img, result, model.CLASSES, score_thr=args.score_thr, wait_time=1, out_file=filepath, show=args.show)


if __name__ == '__main__':
    main()
