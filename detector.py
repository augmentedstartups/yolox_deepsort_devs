import time
import torch
import numpy as np
import cv2
from loguru import logger

from yolox.exp.build import get_exp_by_name
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import postprocess, vis


class Predictor():
    def __init__(self, model='yolox-s', ckpt='yolox_s.pth', visual=True):
        super(Predictor, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.exp = get_exp_by_name(model)
        self.test_size = self.exp.test_size  
        self.model = self.exp.get_model()
        self.model.to(self.device)
        self.model.eval()
        checkpoint = torch.load(ckpt, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.preproc = ValTransform(legacy=False)


    def inference(self, img, visual=True, conf=0.5):
        img_info = {"id": 0}
        
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        img_info['img'] = img
        height, width = img.shape[:2]
        img_info["height"], img_info["width"], img_info["img"]  = height, width, img
        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio
        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        
        if self.device == torch.device('cuda'):
            img = img.cuda()
            img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.exp.num_classes, self.exp.test_conf, self.exp.nmsthre 
                )[0].cpu().numpy()
        
        img_info['boxes'] = outputs[:, 0:4]/ratio
        img_info['scores'] = outputs[:, 4] * outputs[:, 5]
        img_info['class_ids'] = outputs[:, 6]
        img_info['box_nums'] = outputs.shape[0]

        if visual:
            img_info['visual'] = vis(img_info['img'], img_info['boxes'], img_info['scores'], img_info['class_ids'], conf, COCO_CLASSES)

        logger.img_info("Infer time: {:.4f}s".format(time.time() - t0))
        return img_info
    
    
    # def inference(self, img):
    #     img_info = {"id": 0}
    #     if isinstance(img, str):
    #         img_info["file_name"] = os.path.basename(img)
    #         img = cv2.imread(img)
    #     else:
    #         img_info["file_name"] = None

    #     height, width = img.shape[:2]
    #     img_info["height"] = height
    #     img_info["width"] = width
    #     img_info["raw_img"] = img

    #     ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
    #     img_info["ratio"] = ratio

    #     img, _ = self.preproc(img, None, self.test_size)
    #     img = torch.from_numpy(img).unsqueeze(0)
    #     img = img.float()
    #     if self.device == "gpu":
    #         img = img.cuda()
    #         if self.fp16:
    #             img = img.half()  # to FP16

    #     with torch.no_grad():
    #         t0 = time.time()
    #         outputs = self.model(img)
    #         if self.decoder is not None:
    #             outputs = self.decoder(outputs, dtype=outputs.type())
    #         outputs = postprocess(
    #             outputs, self.num_classes, self.confthre,
    #             self.nmsthre, class_agnostic=True
    #         )
    #         logger.info("Infer time: {:.4f}s".format(time.time() - t0))
    #     return outputs, img_info


if __name__=='__main__':
    predictor = Predictor()
    img = cv2.imread('img.jpeg')
    img_,out = predictor.inference(img)
    print(out)