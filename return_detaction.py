from PIL import Image
from ultralytics import YOLOv10
from ultralytics.engine.results import Results
import torch

def detaction(imgpath):
# 테스트할 이미지 경로
  
    # 가중치 파일 경로
    path = './best.pt'

    # YOLOv10 모델 초기화
    model = YOLOv10(path)

    # 옵션: GPU를 사용할 경우 디바이스 설정
    device = 'cuda'  # CPU를 사용하려면 'cpu'로 변경하세요
    model.to(device)

    # 옵션: 신뢰도 임계값 설정
    model.conf = 0.6

    # PIL을 사용하여 이미지 로드
    img = Image.open(imgpath)

    # 이미지를 적절한 크기로 리사이즈
    input_size = (640, 640)
    img_resized = img.resize(input_size, Image.BICUBIC)

    # 예측 수행
    results = model.predict(img_resized, save=False)

    # 각 results 객체에서 names 속성을 추출하여 리스트로 만듭니다.
    labels = [list(result.names.values()) for result in results]

        
    return labels
