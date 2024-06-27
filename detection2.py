from PIL import Image
from ultralytics import YOLOv10
import torch
import json

def detection2(imgpath):
    # 가중치 파일 경로
    path = './runs/detect/train6/weights/best.pt'

    # YOLOv10 모델 초기화
    model = YOLOv10(path)

    # 옵션: GPU를 사용할 경우 디바이스 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # 옵션: 신뢰도 임계값 설정
    model.conf = 0.6

    # PIL을 사용하여 이미지 로드
    img = Image.open(imgpath)

    # 이미지를 적절한 크기로 리사이즈
    input_size = (640, 640)
    img_resized = img.resize(input_size, Image.BICUBIC)

    # 예측 수행
    results = model.predict(img_resized, save=True)

    # 결과 객체가 리스트인지 확인
    if isinstance(results, list):
        results = results[0]

    # 클래스 인덱스 추출 및 정수로 변환
    class_indices = [int(box.cls.item()) for box in results.boxes]

    # 클래스 인덱스를 사용하여 클래스 이름 출력
    class_names = [results.names[index] for index in class_indices]

    detected_objects = {
        "detected_objects": class_names
    }

    # 결과 출력
    return json.dumps(detected_objects, ensure_ascii=False)
