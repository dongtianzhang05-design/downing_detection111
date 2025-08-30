from ultralytics import YOLO
# 加载训练好的模型，改为自己的路径
model = YOLO('/autodl-nas/ultralytics-8.3.20/runs/detect/train8/weights/best.pt')  #修改为训练好的路径
source = '/autodl-nas/ultralytics-8.3.20/img_6.png' #修改为自己的图片路径及文件名
# 运行推理，并附加参数
model.predict(source, save=True)