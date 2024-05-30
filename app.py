import torch
from torchvision import transforms
from PIL import Image
import gradio as gr
import numpy as np


class Cultural_relicscls:
    def __init__(self, model_pth='place365.pth'):
        # 初始化模型
        network = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')
        network.fc = torch.nn.Linear(512, 365)
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

        # 初始化类别名称
        self.id2categories_name = dict()
        with open('categories_places365.txt') as f:
            lines = f.readlines()
            for line in lines:
                category_name, id = line.split(' ')
                id = int(id)
                category_name = category_name[3:]
                self.id2categories_name[id] = category_name

    def predict(self, img, top_n=5):
        # 如果图像是numpy.ndarray类型，将其转换为PIL Image
        if isinstance(img, np.ndarray):
            img = Image.fromarray(np.uint8(img))
        img = self.transform(img)
        img = img.unsqueeze(0)
        out = self.model(img)
        out = torch.nn.functional.softmax(out, dim=1)
        score, label = torch.topk(out[0], k=5)
        return {self.id2categories_name[int(c)]: float(s) for c, s in zip(label, score)}


"""
        label = np.argsort(out)[-top_n:][::-1]
        top_labels=dict()
        for c in label:
            c=int(c)
            c_name=self.id2categories_name[c]
            score=float(out[c])
            top_labels[c_name]=score
        return top_labels
"""

cls_model = Place365cls()


def predict(image):
    result = cls_model.predict(image, top_n=5)
    return result


if __name__ == "__main__":
    interface = gr.Interface(fn=predict, inputs="image",
                             outputs=gr.Label(),
                             title="Place365 classification"
                             )
    interface.launch()
