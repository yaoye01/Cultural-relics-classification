import torch
import torch.nn as nn
from torchvision import transforms, datasets
import pathlib
import warnings
import os

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 数据集目录路径
data_dir = './RelicDatabase/images'

# 数据预处理
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为 224x224 像素
    transforms.ToTensor(),          # 将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化图像像素值
])

# 加载数据集
dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)

# 创建标签列表
labels = []
for image_path, label in dataset.samples:
    class_name = os.path.basename(os.path.dirname(image_path))
    info_file = os.path.join(os.path.dirname(image_path), 'information.txt')
    with open(info_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        period = lines[0].split('：')[1].strip()  # 获取时期
        name = lines[1].split('：')[1].strip()    # 获取名称
        label_str = f"{period}{name}"
        labels.append(label_str)

'''
# 打印前几个图像的标签
for i in range(10):
    print(f"图像 {i+1} 的标签：{labels[i]}")
'''

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
print(train_dataset, test_dataset)

batch_size = 4
train_dl = torch.utils.data.DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=1,
                                       pin_memory=False)
test_dl = torch.utils.data.DataLoader(test_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=1,
                                      pin_memory=False)
