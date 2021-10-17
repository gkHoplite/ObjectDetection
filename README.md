# 설명 
- TorchVision에서 가져온 코드로 자동차 번호판 검사, 마스크 착용/미착용 검사에 사용하였습니다. [TorchVision](https://github.com/pytorch/vision) , [Yolo v5](https://github.com/ultralytics/yolov5)
- 마스크 착용/미착용을 구분하여 ObjectTracking을 실시하였습니다. Yolo-v5에서 모델을 훈련시켜 DeepSort에 적용하였습니다. [DeepSort(yolov5기반으로 작동)](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)

- 이 Repository는 Object Detection Code만 포함하고 있으며 훈련결과 pt파일을 포함하고 있지 않습니다.

# 사용법
- 설치 [TorchVision](https://github.com/pytorch/vision)의 python setup.py install
- "train.py --visualize-only --data-path C:/파일경로"를 설정하여 훈련한 모델을 사용해볼 수 있습니다.
- 그 외 다양한 변수 설정은 train.py에 있는 def get_args_parser(add_help=True)에서 확인할 수 있습니다.

# 활용
- Kaggle 마스크 데이터셋과 Youtube에 있는 거리 촬영영상을 Cvat에서 Labeling하여 데이터를 구축하였습니다.
- Object Tracking에서 Faster-RCNN 모델은 처리속도가 너무 느려 의미가 없다고 판단하였고 Yolo-v5만 테스트하였습니다.
- mAP .5:.95기준 (mAP 0.5부터 0.05씩 더하여 0.95까지의 총 합을 평균한 데이터, 이때 IOU>.75, NMS(Thresh hold>0.5)


# 개선할 점
- Faster-RCNN이 Object Tracking에서 처리가 느린 이유 찾아보기
  1. Yolo v5는 hidden layer의 복잡한 정도에 따라 5단계가 있다. 
  2. 가장 무거운 5단계 Yolo보다 Faster-RCNN모델이 2배나 더 무겁다.

- 그래픽카드의 용량이 작아 배치 사이즈를 8이상으로 설정할 수 없음. 배치 사이즈에 따라 손실함수가 다르게 갱신되므로 배치 사이즈를 변경하여 정확도를 비교해볼 수 있음
