from ultralytics import YOLO
model = YOLO("/autodl-nas/ultralytics-8.3.20/runs/detect/train9/weights/last.pt")
metrics = model.val(data="data.yaml", split="test",save=True)  # 在 test 集合上评估
print(metrics)

