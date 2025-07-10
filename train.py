import warnings, os

warnings.filterwarnings('ignore')
from ultralytics import RTDETR



if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/ed-detr/ED-Detr.yaml')
    # model.load('runs/train/rtdetr-dee-detr32/weights/last.pt') # loading pretrain weights
    model.train(data='ultralytics/cfg/datasets/RSOD.yaml',
                cache=False,
                imgsz=640,
                epochs=260,
                batch=8,
                workers=8, 
                device='0', 
                # resume=True, # last.pt path
                project='runs/train',
                )

