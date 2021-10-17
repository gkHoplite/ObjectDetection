# 설명 
- TorchVision에서 가져온 코드로 자동차 번호판 검사, 마스크 착용/미착용 검사에 사용하였습니다. [TorchVision](https://github.com/pytorch/vision) , [Yolo v5](https://github.com/ultralytics/yolov5)
- 마스크 착용/미착용을 구분하여 ObjectTracking을 실시하였습니다. Yolo-v5에서 모델을 훈련시켜 DeepSort에 적용하였습니다. [DeepSort(yolov5기반으로 작동)](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)

- 이 Repository는 Object Detection Code만 포함하고 있으며 훈련결과 pt파일을 포함하고 있지 않습니다.

# 사용법
- 설치 [TorchVision](https://github.com/pytorch/vision)의 python setup.py install
- "train.py --visualize-only --data-path C:/파일경로"를 설정하여 훈련한 모델을 사용해볼 수 있습니다.
- 그 외 다양한 변수 설정은 train.py에 있는 def get_args_parser(add_help=True)에서 확인할 수 있습니다.

# 활용
- Kaggle 마스크 데이터셋과 Youtube에 있는 거리 촬영영상을 Cvat에서 Labeling하여 데이터를 구축.
- Yolov5와 Faster-Rcnn을 비교.
- mAP .5:.95기준 (mAP 0.5부터 0.05씩 더하여 0.95까지의 총 합을 평균한 데이터, 이때 IOU>.75, NMS(Thresh hold>0.5)
