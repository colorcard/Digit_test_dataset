import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ========================
# 🔧 配置区域
# ========================
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
LEARNING_RATE = 1e-3
EPOCHS = 10
MODEL_PATH = "../pth/mnist_best_cnn.pth"

# 自动选择设备
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用设备: {device}")

# ========================
# 📦 数据处理
# ========================
transform = transforms.Compose([
    transforms.RandomRotation(10),  # 数据增强
    transforms.ToTensor()
])

train_data = datasets.MNIST(root="../data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="../data", train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE)

# ========================
# 🧠 模型定义（高性能 CNN）
# ========================
class BestMNISTModel(nn.Module):
    def __init__(self):
        super(BestMNISTModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)

# model = BestMNISTModel().to(device)

# ========================
# 🧠 ResNet 模型定义（简化版适用于 MNIST）
# ========================
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNetMNIST(nn.Module):
    def __init__(self):
        super(ResNetMNIST, self).__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 64, blocks=2)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 10)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [BasicBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = ResNetMNIST().to(device)

# ========================
# ⚙️ 损失函数与优化器
# ========================
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)



# ========================
# 🏋️ 训练函数
# ========================
def train(epoch):
    model.train()
    running_loss = 0
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch % 100 == 0:
            print(f"Epoch {epoch} [{batch * len(X)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    return avg_loss

# ========================
# 🧪 测试函数
# ========================
def test(epoch):
    model.eval()
    correct = 0
    total_loss = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = loss_fn(output, y)
            total_loss += loss.item() * y.size(0)
            pred = output.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = 100. * correct / total
    avg_loss = total_loss / total
    print(f"\n✅ 测试准确率: {acc:.2f}%, 测试损失: {avg_loss:.4f}\n")

    return avg_loss, acc

# ========================
# 🚀 主训练循环
# ========================
for epoch in range(1, EPOCHS + 1):
    train_loss = train(epoch)
    test_loss, test_acc = test(epoch)

# ========================
# 💾 保存模型
# ========================
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ 模型参数已保存至: {MODEL_PATH}")