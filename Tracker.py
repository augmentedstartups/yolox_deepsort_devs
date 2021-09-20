import sys
sys.path.insert(0, './YOLOX')
from yolox.data.datasets.coco_classes import COCO_CLASSES
from detector import Predictor
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2
from yolox.utils import vis
from yolox.utils.visualize import vis_track

from yolox.exp import get_exp
import numpy as np


class_names = COCO_CLASSES
TRAIL_LEN = 64

from collections import deque



class Tracker():
    def __init__(self, filter_class=None, model='yolox-s', ckpt='yolox_s.pth'):
        self.detector = Predictor(model, ckpt)
        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
        self.filter_class = filter_class
    def update(self, image):
        _,info = self.detector.inference(image, visual=False)
        outputs = []
        if info['box_nums']>0:
            bbox_xywh = []
            scores = []
            #bbox_xywh = torch.zeros((info['box_nums'], 4))
            for [x1, y1, x2, y2], class_id, score  in zip(info['boxes'],info['class_ids'],info['scores']):
                if self.filter_class and class_names[int(class_id)] not in self.filter_class:
                    continue
                # color = compute_color_for_labels(int(class_id))
                bbox_xywh.append([int((x1+x2)/2), int((y1+y2)/2), x2-x1, y2-y1])                
                scores.append(score)
                
            bbox_xywh = torch.Tensor(bbox_xywh)
            outputs = self.deepsort.update(bbox_xywh, scores, image)
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -2]
                object_id = outputs[:, -1]
                # draw_boxes(image, bbox_xyxy, object_id,identities)
                image = self.visual(image, outputs)

        return image, outputs

    def visual(self, img, output, cls_conf=0.35):
        ratio = min(self.detector.test_size[0] / img.shape[0], self.detector.test_size[0] / img.shape[1])
        if output is None:
            return img
        # output = output.cpu()
        bboxes = output[:, 0:4]
        # preprocessing: resize
        bboxes /= ratio
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


if __name__=='__main__':
    tracker = Tracker(filter_class=None, model='yolox-s', ckpt='/content/yolox_deepsort_devs/weights/yolox_s.pth')    # instantiate Tracker

    cap = cv2.VideoCapture('video.mp4') 
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)

    vid_writer = cv2.VideoWriter(
        'video_tracked.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    ) # open one video
    frame_count = 0
    while True:
        ret_val, frame = cap.read() # read frame from video
        
        if ret_val:
            img_visual, bbox = tracker.update(frame)  # feed one frame and get result
            vid_writer.write(img_visual)
            if frame_count == 500:
                break
            frame_count +=1
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break

    cap.release()
    vid_writer.release()
    cv2.destroyAllWindows()
