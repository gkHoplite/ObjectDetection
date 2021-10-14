# 설명 
- Object Detection, TorchVision에서 가져온 코드를 마스크 착용/미착용 검사에 사용하였습니다. [TorchVision](https://github.com/pytorch/vision)
- ObjectTracking을 사용하기 원한다면 -> [DeepSort](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)를 사용하였습니다.
- 이 Repository는 Object Detection Code만 포함하고 있습니다.

# 훈련
- Kaggle 마스크 데이터셋과 Youtube에 있는 거리 촬영영상을 Cvat에서 Labeling하여 데이터를 구축하였습니다.
- Yolov5와 Faster-Rcnn을 비교
- mAP .5:.95 (mAP 0.5부터 0.05씩 더하여 0.95까지의 총 합을 평균한 데이터, 이때 IOU>.75, NMS(IOU>0.5, Thresh hold>0.5)

# 사용법
- 설치 [TorchVision](https://github.com/pytorch/vision)의 python setup.py install
- "train.py --visualize-only --data-path C:/파일경로"를 설정하여 훈련한 모델을 사용해볼 수 있습니다.
- 그 외 다양한 변수 설정은 train.py에 있는 def get_args_parser(add_help=True)에서 확인할 수 있습니다.

# Object detection reference training scripts

- Exclusively-Dark-Image-Dataset [Github url](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/tree/master/Dataset)
    - [Images](https://drive.google.com/file/d/1GZqHFzTLDI-1rcOctHdf-c16VgagWocd/view)
    - [Annotation](https://drive.google.com/file/d/1goqzN0Eg7YqClZfP3cQ9QjENFrEhildz/view)


This folder contains reference training scripts for object detection.
They serve as a log of how to train specific models, to provide baseline
training and evaluation scripts to quickly bootstrap research.

To execute the example commands below you must install the following:

```
cython
pycocotools
matplotlib
```

You must modify the following flags:

`--data-path=/path/to/coco/dataset`

`--nproc_per_node=<number_of_gpus_available>`

Except otherwise noted, all models have been trained on 8x V100 GPUs. 

### Faster R-CNN ResNet-50 FPN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model fasterrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```

### Faster R-CNN MobileNetV3-Large FPN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model fasterrcnn_mobilenet_v3_large_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```

### Faster R-CNN MobileNetV3-Large 320 FPN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model fasterrcnn_mobilenet_v3_large_320_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```

### RetinaNet
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model retinanet_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3 --lr 0.01
```

### SSD300 VGG16
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model ssd300_vgg16 --epochs 120\
    --lr-steps 80 110 --aspect-ratio-group-factor 3 --lr 0.002 --batch-size 4\
    --weight-decay 0.0005 --data-augmentation ssd
```

### SSDlite320 MobileNetV3-Large
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model ssdlite320_mobilenet_v3_large --epochs 660\
    --aspect-ratio-group-factor 3 --lr-scheduler cosineannealinglr --lr 0.15 --batch-size 24\
    --weight-decay 0.00004 --data-augmentation ssdlite
```


### Mask R-CNN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model maskrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```


### Keypoint R-CNN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco_kp --model keypointrcnn_resnet50_fpn --epochs 46\
    --lr-steps 36 43 --aspect-ratio-group-factor 3
```

