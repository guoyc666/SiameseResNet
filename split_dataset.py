import os
import random
import shutil
from pathlib import Path

# 设置路径
data_dir = Path("../train")  # 数据根目录

dst_dir = Path("../data")  # 数据根目录
train_dir = dst_dir / "train"
valid_dir = dst_dir / "valid"
test_dir = dst_dir / "test"

train_dir.mkdir(parents=True, exist_ok=True)
valid_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

# 划分比例
train_ratio = 0.8
test_area = (987, 1080)

# 遍历每个类别文件夹
for class_folder in data_dir.iterdir():
    if class_folder.is_dir():  # 
        images = list(class_folder.glob("*.png"))
        random.shuffle(images)

        if int(class_folder.name.split('_')[1]) < 987 :
            split_idx = int(len(images) * train_ratio)
            train_images = images[:split_idx]
            test_images = images[split_idx:]

            # 创建对应的类别文件夹
            (train_dir / class_folder.name).mkdir(parents=True, exist_ok=True)
            (valid_dir / class_folder.name).mkdir(parents=True, exist_ok=True)

            # 移动图像文件到新的目录
            for img_path in train_images:
                shutil.copy(img_path, train_dir / class_folder.name / img_path.name)
            for img_path in test_images:
                shutil.copy(img_path, valid_dir / class_folder.name / img_path.name)

        else:
            (test_dir / class_folder.name).mkdir(parents=True, exist_ok=True)
            for img_path in images:
                shutil.copy(img_path, test_dir / class_folder.name / img_path.name)

print("数据集划分完成！")
