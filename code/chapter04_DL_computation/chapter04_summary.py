import torch
from torch import nn

# 1. 构建模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)
    def forward(self, x):
        return self.output(self.act(self.hidden(x)))

# 2. 参数初始化
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

net = MLP()
net.apply(init_weights)

# 3. 自定义层
class CenteredLayer(nn.Module):
    def forward(self, x):
        return x - x.mean()

# 4. 保存与加载
torch.save(net.state_dict(), 'mlp_params.pt')
net2 = MLP()
net2.load_state_dict(torch.load('mlp_params.pt'))

# 5. GPU 加速
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net2.to(device)
x = torch.rand(2, 784).to(device)
y = net2(x) 