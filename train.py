from datasets import *
from DPN import DPN
import copy
import matplotlib.pyplot as plt
import warnings


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


model = DPN92().to(device)


# 训练循环
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # 训练集的大小
    num_batches = len(dataloader)  # 批次数目, (size/batch_size，向上取整)

    train_loss, train_acc = 0, 0  # 初始化训练损失和正确率

    for x, y in dataloader:  # 获取图片及其标签
        x, y = x.to(device), y.to(device)

        # 计算预测误差
        pred = model(x)  # 网络输出
        loss = loss_fn(pred, y)  # 计算网络输出pred和真实值y之间的差距，y为真实值，计算二者差值即为损失

        # 反向传播
        optimizer.zero_grad()  # grad属性归零
        loss.backward()  # 反向传播
        optimizer.step()  # 每一步自动更新

        # 记录acc与loss
        train_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
        train_loss += loss.item()

    train_acc /= size
    train_loss /= num_batches

    return train_acc, train_loss


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)  # 训练集的大小
    num_batches = len(dataloader)  # 批次数目, (size/batch_size，向上取整)
    test_loss, test_acc = 0, 0  # 初始化测试损失和正确率

    # 当不进行训练时，停止梯度更新，节省计算内存消耗
    # with torch.no_grad():
    for imgs, target in dataloader:  # 获取图片及其标签
        with torch.no_grad():
            imgs, target = imgs.to(device), target.to(device)

            # 计算误差
            tartget_pred = model(imgs)  # 网络输出
            loss = loss_fn(tartget_pred, target)  # 计算网络输出和真实值之间的差距，targets为真实值，计算二者差值即为损失

            # 记录acc与loss
            test_loss += loss.item()
            test_acc += (tartget_pred.argmax(1) == target).type(torch.float).sum().item()

    test_acc /= size
    test_loss /= num_batches

    return test_acc, test_loss


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()  # 创建损失函数

epochs = 40

train_loss = []
train_acc = []
test_loss = []
test_acc = []

best_acc = 0  # 设置一个最佳准确率，作为最佳模型的判别指标

if __name__ == '__main__':
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_train_acc, epoch_train_loss = train(train_dl, model, loss_fn, optimizer)
        # scheduler.step() #更新学习率（调用官方动态学习率接口时使用）

        model.eval()
        epoch_test_acc, epoch_test_loss = test(test_dl, model, loss_fn)

        # 保存最佳模型到best_model
        if epoch_test_acc > best_acc:
            best_acc = epoch_test_acc
            best_model = copy.deepcopy(model)

        train_acc.append(epoch_train_acc)
        train_loss.append(epoch_train_loss)
        test_acc.append(epoch_test_acc)
        test_loss.append(epoch_test_loss)

        # 获取当前的学习率
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        template = ('Epoch: {:2d}. Train_acc: {:.1f}%, Train_loss: {:.3f}, Test_acc:{:.1f}%, Test_loss:{:.3f}, Lr: {:.2E}')
        print(
            template.format(epoch + 1, epoch_train_acc * 100, epoch_train_loss, epoch_test_acc * 100, epoch_test_loss, lr))

    PATH = './model.pth'
    torch.save(model.state_dict(), PATH)

    print('Done')

    warnings.filterwarnings("ignore")  # 忽略警告信息
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.rcParams['figure.dpi'] = 100  # 分辨率

    epochs_range = range(epochs)

    plt.figure(figsize=(12, 3))
    plt.subplot(1, 2, 1)

    plt.plot(epochs_range, train_acc, label='Training Accuracy')
    plt.plot(epochs_range, test_acc, label='Test Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss, label='Training Loss')
    plt.plot(epochs_range, test_loss, label='Test Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
