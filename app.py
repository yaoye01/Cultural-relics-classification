import torch
from torchvision import transforms
from PIL import Image
import gradio as gr
import numpy as np
from DPN import DPN


def DPN92(n_class=35):
    cfg = {
        "group": 32,
        "in_channel": 64,
        "mid_channels": (96, 192, 384, 768),
        "out_channels": (256, 512, 1024, 2048),
        "dense_channels": (16, 32, 24, 128),
        "num": (3, 4, 20, 3),
        "classes": n_class
    }
    return DPN(cfg)


# 初始化类别名称
labels_list = []  # 创建一个空列表来存储 labels
with open('labels.txt', 'r', encoding='gbk') as f:
    for line in f:
        label = line.strip()  # 去除行首尾的空白字符
        labels_list.append(label)  # 将 label 添加到列表中


class Cultural_relicscls:
    def __init__(self, model_pth='model.pth'):
        # 初始化模型
        network = DPN92()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        network.load_state_dict(torch.load(model_pth, map_location=device))
        self.model = network
        self.model.eval()

        # 初始化数据预处理
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[127.5], std=[127.5])
        ])

    def predict(self, img, top_n=5):
        # 如果图像是numpy.ndarray类型，将其转换为PIL Image
        if isinstance(img, np.ndarray):
            img = Image.fromarray(np.uint8(img))
        img = self.transform(img)
        img = img.unsqueeze(0)
        out = self.model(img)
        out = torch.nn.functional.softmax(out, dim=1)
        score, label = torch.topk(out[0], k=5)
        return {labels_list[int(c)]: float(s) for c, s in zip(label, score)}


cls_model = Cultural_relicscls()


def predict(image):
    result = cls_model.predict(image, top_n=5)
    return result


if __name__ == "__main__":
    interface = gr.Interface(fn=predict, inputs="image",
                             outputs=gr.Label(),
                             title="Cultural relics classification"
                             )
    interface.launch()
