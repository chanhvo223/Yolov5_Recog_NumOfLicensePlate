# Yolov5_Recog_NumOfLicensePlate
Tham khảo:
- Cách làm: https://www.youtube.com/watch?v=-3i9AEOjmrk&ab_channel=Thi%E1%BB%87nD%C6%B0%C6%A1ng
- Giải thích: https://www.youtube.com/watch?v=eSS0EnCX1A0&ab_channel=Lato%27channel
- Colab custom: https://colab.research.google.com/github/roboflow-ai/yolov5-custom-training-tutorial/blob/main/yolov5-custom-training.ipynb


1. Liên Kết với GG Driver
-------------------------------
from google.colab import drive
import os
drive.mount('/content/drive/')
os.chdir('drive/MyDrive/')
-------------------------------
os.getcwd() 	#Xem vị trí hiện tại
-------------------------------


2 Setup Yolov5
Link: https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb#scrollTo=wbvMlHd_QwMG
-------------------------------
#!git clone https://github.com/ultralytics/yolov5  # clone
%cd yolov5
#%pip install -qr requirements.txt comet_ml  # install
import torch
import utils
display = utils.notebook_init()  # checks
-------------------------------


3. Cài zip và unrar trên driver
-------------------------------
!pip install unrar
!unrar x /content/drive/MyDrive/Study/MachineLearning/BienSo/dataset.rar
-------------------------------


4. Thêm data đã unrar/unzip vào file yolov5
-------------------------------
*Cách 1: Sử dụng mô hình chuẩn theo file coco128.yaml *
- git: https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml
- path: ../datasets/coco128
- train: images/train2017
- val: images/train2017
-> mô hình coco128 --> có 2 file images và labels --> trong đó có file train2017
*Cách 2: Vào file cocp128.yaml thay đổi đường dẫn như dưới*
- path: /content/yolov5/dataset
- train: /content/yolov5/dataset/images
- val: /content/yolov5/dataset/images
- class
Link tham khảo: https://www.youtube.com/watch?v=-3i9AEOjmrk&ab_channel=Thi%E1%BB%87nD%C6%B0%C6%A1ng
*Cách 3: Up data lên roboflow và chạy câu lệnh để up để data vào colab
-------------------------------
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="Cny6v5mbH0HFzQik9WEL")
project = rf.workspace("vsk-9bee1").project("platevehiclevn")
dataset = project.version(1).download("yolov5")
-------------------------------


5. Train YoLov5 model
- epochs là số lần train --> càng nhiều thì càng chính xác
- Sau khi train xong thì có file runs/train/exp/weights --> best.pt & last.pt
-------------------------------
!python train.py --img 640 --batch 16 --epochs 5 --data coco128.yaml --weights yolov5s.pt --cache
-------------------------------


6. Train tiếp bằng file best.pt
- Thay file yolo5x.pt bằng file best.pt bằng link đường dẫn vào file best.pt
-------------------------------
!python train.py --img 640 --batch 16 --epochs 50 --data coco128.yaml --weights /content/yolov5/runs/train/exp/weights/best.pt --cache
-------------------------------


7. Detect data
-------------------------------
*Cách 1: Chạy trên máy tính local
Thay đổi model đã train trong file detect.py --> line 244 --> sửa thành best.pt
Tải file model best.pt và file detect.py về máy
Tải git hub yolov5 về máy
Chuyển 2 file trên vào dự án github đó
Mở file detect.py và chạy: !python detect.py --weights best.pt --img 640 --conf 0.1 --source (pathfolder, img, 0: webcam)
*Cách 2: Chạy trên gg colab
-------------------------------
!python detect.py --weights /content/drive/MyDrive/yolov5/runs/train/exp/weights/best.pt --img 640 --conf 0.1 --source (pathfolder, img, 0: webcam)
-------------------------------


8. Crop ảnh 
- Thay đổi trong file detect.py --> Line 69: --save-crop = True
- Đặt ảnh vào file data/images
-------------------------------
!python /content/drive/MyDrive/yolov5/detect.py --save-crop
-------------------------------
