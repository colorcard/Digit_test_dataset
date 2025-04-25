import os
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

# ========================
# 🔧 配置区域
# ========================
MODEL_PATH = "../pth/mnist_best_cnn.pth"
DATA_PATH = "../test"  # 图片所在路径
BATCH_SIZE = 1000
SUBMIT_CSV = "../submission.csv"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备：{device}")

# ========================
# 🧠 模型结构定义
# ========================
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
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

def load_model(path=MODEL_PATH):
    # model = CNNModel()
    model = ResNetMNIST()

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model



# ========================
# 🖼️ 自定义数据集（仅读取图片）
# ========================
class ImageOnlyDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_files = sorted([
            f for f in os.listdir(img_dir) if f.endswith(".png")
        ], key=lambda x: int(os.path.splitext(x)[0]))
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        file_name = self.img_files[idx]
        img_id = int(os.path.splitext(file_name)[0])
        image = Image.open(os.path.join(self.img_dir, file_name)).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, img_id

# ========================
# 🧪 执行预测 + 导出 CSV
# ========================
def predict_and_export(model):
    transform = transforms.ToTensor()
    dataset = ImageOnlyDataset(DATA_PATH, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    predictions = []

    with torch.no_grad():
        for X, ids in loader:
            X = X.to(device)
            outputs = model(X)
            preds = outputs.argmax(1).cpu().numpy()
            predictions.extend(zip(ids.numpy(), preds))

    df = pd.DataFrame(predictions, columns=["id", "label"])
    df = df.sort_values("id")  # 按 id 排序
    df.to_csv(SUBMIT_CSV, index=False)
    print(f"✅ 提交文件已导出：{SUBMIT_CSV}")

# ========================
# 🚀 主程序入口
# ========================
if __name__ == "__main__":
    model = load_model()
    predict_and_export(model)