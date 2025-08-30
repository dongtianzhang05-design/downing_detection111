import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
  model = YOLO('ultralytics/cfg/models/11/yolo11n.yaml')#新建一个 YOLO 模型，结构基于 yolo11n.yaml 配置（轻量级版本，适合小数据集/小显存）
  model.load('yolo11n.pt')  #注释则不加载//加载预训练权重
#   results = model.train(
#     data='data.yaml',  #数据集配置文件的路径
#     epochs=200,  #训练轮次总数
#     batch=16,  #批量大小，即单次输入多少图片训练
#     imgsz=640,  #训练图像尺寸
#     workers=8,  #加载数据的工作线程数
#     device= 0,  #指定训练的计算设备，无nvidia显卡则改为 'cpu'
#     optimizer='SGD',  #训练使用优化器，可选 auto,SGD,Adam,AdamW 等
#     amp= True,  #True 或者 False, 解释为：自动混合精度(AMP) 训练
#     cache=False  # True 在内存中缓存数据集图像，服务器推荐开启
# )
  #class_weights = [0.1578978413830878, 0.26801679704506837, 0.5740853615718439]
  results = model.train(
      data='data.yaml',
      epochs=300,  # 建议先 100，看验证集表现，再决定是否加到 200
      batch=32,  # 显存允许的话 32 比 16 更稳
      imgsz=768,  # 小数据集 640 就够了，不必开太大
      workers=8,  # 8 线程加载数据就行，别开太大
      device=0,  # GPU 0
      optimizer='AdamW',  # 小数据集用 AdamW 收敛更快
      amp=True,  # 开启混合精度，省显存提速
      cache=True # 服务器内存一般够，开启加速加载
  )
