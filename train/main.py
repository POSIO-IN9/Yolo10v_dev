import subprocess
import argparse
import torch
import logging
from ultralytics import YOLO


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    return logger

def main(opt):
    logger = setup_logging()

    # YOLO 모델 초기화
    model = YOLO(opt.model)
    logger.info(f"YOLO 모델 초기화 완료. 모델 설정 파일 경로: {opt.model}")

    # Check CUDA availability and select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"사용할 장치: {device}")

    # Log input arguments
    logger.info(f"인수: imgsz={opt.imgsz}, batch={opt.batch}, epochs={opt.epochs}, data={opt.data}, workers={opt.workers}, amp={opt.amp}")

    # Execute the command
    command = [
        "yolo", "train",
        f"model={opt.model}",
        f"imgsz={opt.imgsz}",
        f"batch={opt.batch}",
        f"epochs={opt.epochs}",
        f"data={opt.data}",
        f"workers={opt.workers}",
    ]

    if opt.amp:
        command.append("amp=true")

    # Create subprocess.Popen to execute the command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Read stdout and stderr line by line and print in real-time
    while True:
        output = process.stdout.readline().strip()
        error = process.stderr.readline().strip()

        if output:
            print(output)
        if error:
            print(error)

        # Check if the process has finished
        if process.poll() is not None:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgsz', type=int, default=640, help='이미지 크기')
    parser.add_argument('--batch', type=int, default=16, help='배치 크기')
    parser.add_argument('--epochs', type=int, default=30, help='에폭 수')
    parser.add_argument('--data', type=str, default='farmpj.yaml', help='데이터셋 YAML 파일 경로')
    parser.add_argument('--model', type=str, default='./train/yolov10n.yaml', help='YOLO 모델 설정 파일 경로')
    parser.add_argument('--amp', action='store_true', help='자동 혼합 정밀도 사용 여부')
    parser.add_argument('--workers', type=int, default=0, help='데이터 로딩을 위한 워커 수')
    opt = parser.parse_args()

    main(opt)
