import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# 读取数据
data = pd.read_csv(r'D:\code\ai+\test\data\amzn-anon-access-samples\amzn-anon-access-samples-history-2.0.csv')

# 数据预处理
data['REQUEST_DATE'] = pd.to_datetime(data['REQUEST_DATE'], errors='coerce')
data['AUTHORIZATION_DATE'] = pd.to_datetime(data['AUTHORIZATION_DATE'], errors='coerce')

# 删除包含 NaT 的行
data = data.dropna(subset=['REQUEST_DATE', 'AUTHORIZATION_DATE'])

data['AUTHORIZATION_TIME'] = (data['AUTHORIZATION_DATE'] - data['REQUEST_DATE']).dt.total_seconds()

# 对 ACTION 进行编码
data['ACTION_CODE'] = data['ACTION'].map({'add_access': 1, 'remove_access': 0})

# 准备数据
X = data['ACTION_CODE'].values
y = data['AUTHORIZATION_TIME'].values

# 数据标准化
scaler = StandardScaler()
y = scaler.fit_transform(y.reshape(-1, 1)).flatten()

# 转换为 PyTorch 张量
X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 创建数据集和数据加载器
dataset = CustomDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 定义线性回归模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

# 初始化模型、损失函数和优化器
model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    scheduler.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 打印模型参数
print("模型参数:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
    