"""
FashionMNIST + LeNet 训练与预测完整示例
--------------------------------------
本文件演示如何用 PyTorch 在 FashionMNIST 数据集上训练 LeNet 卷积神经网络，并进行预测。
适合初学者理解深度学习的完整工程流程。
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 数据加载与预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为张量并归一化到[0,1]
])
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128)

# 2. LeNet 模型定义
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), nn.Sigmoid(), nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5), nn.Sigmoid(), nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)

# 3. 训练与评估函数
def train(model, train_loader, test_loader, device, epochs=5):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, n = 0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            total_correct += (out.argmax(1) == y).sum().item()
            n += y.size(0)
        print(f'Epoch {epoch+1}: loss={total_loss/n:.4f}, train acc={total_correct/n:.3f}')
        # 测试集评估
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        print(f'  Test acc: {correct/total:.3f}')

# 4. 预测函数
def predict(model, data_loader, device, class_names):
    model.eval()
    X, y = next(iter(data_loader))
    X, y = X.to(device), y.to(device)
    with torch.no_grad():
        out = model(X)
        pred = out.argmax(1)
    print('真实标签:', [class_names[i] for i in y[:10].cpu().numpy()])
    print('预测标签:', [class_names[i] for i in pred[:10].cpu().numpy()])

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names = train_data.classes
    model = LeNet()
    train(model, train_loader, test_loader, device, epochs=5)
    predict(model, test_loader, device, class_names) 