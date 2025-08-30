import os
from collections import Counter
import numpy as np

# ----------------------------
# 配置：标签目录 & 类别数
# ----------------------------
label_dir = "/autodl-nas/ultralytics-8.3.20/VOCdevkit/train/labels"  # 修改为你的训练标签路径
num_classes = 3  # 修改为你的类别总数


# ----------------------------
# 统计每个类别的目标数量
# ----------------------------
def count_class_instances(label_dir, num_classes=None):
    counter = Counter()

    for file in os.listdir(label_dir):
        if not file.endswith(".txt"):
            continue
        with open(os.path.join(label_dir, file), "r") as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    counter[class_id] += 1

    # 补全类别（没有样本的类别数量设为 0）
    if num_classes is not None:
        for i in range(num_classes):
            counter.setdefault(i, 0)

    return counter


# 统计
class_counts = count_class_instances(label_dir, num_classes)
print("类别统计：", class_counts)

# ----------------------------
# 计算类别权重
# ----------------------------
counts = np.array([class_counts[i] for i in range(num_classes)], dtype=float)

# 方法1：mean / n_i
weights = counts.mean() / counts

# 方法2：可选归一化，保证权重和为1
weights = weights / weights.sum()

# 转为列表，方便在 YOLOv11 里使用
weights_list = weights.tolist()
print("类别权重：", weights_list)

# ----------------------------
# 输出提示
# ----------------------------
print("\n说明：类别权重数组顺序对应 class_id，从0到{}，少数类权重最大。".format(num_classes - 1))
