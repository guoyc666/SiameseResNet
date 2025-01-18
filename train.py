# 完全训练一遍，很烂。废弃
# 由于其他文件修改，不一定能运行
import torch
import torch.optim as optim
from torch.functional import F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from tqdm import tqdm
import random
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SiameseResNet
from test import eval

class SiameseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 收集所有图像路径和标签
        for label in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, label)
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(label)

        self.label_to_idx = {label: idx for idx, label in enumerate(set(self.labels))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img1_path = self.image_paths[index]
        label1 = self.labels[index]
        label1_idx = self.label_to_idx[label1]

        # 随机选择正对或负对
        if random.random() > 0.5:
            # 正对
            img2_path = img1_path  # 选择相同图像
            label2_idx = label1_idx
        else:
            # 负对
            img2_path = img1_path
            while True:
                random_index = random.randint(0, len(self.image_paths) - 1)
                label2 = self.labels[random_index]
                label2_idx = self.label_to_idx[label2]
                if label1 != label2:  # 确保选择不同类别
                    img2_path = self.image_paths[random_index]
                    break

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(1 if label1_idx == label2_idx else 0)  # 1为正对，0为负对

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def contrastive_loss(output1, output2, label, margin=1.0):
    # 计算欧几里得距离
    euclidean_distance = F.pairwise_distance(output1, output2)
    loss = (1 - label) * torch.pow(euclidean_distance, 2) + \
           (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2)
    return loss.mean()

def train(model, train_loader, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for img1, img2, label in tqdm(train_loader, desc=f'[{epoch+1}/{num_epochs}]'):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            optimizer.zero_grad()
            output1, output2 = model(img1, img2)

            loss = contrastive_loss(output1, output2, label) 
            loss.backward() 
            optimizer.step()

        # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
        score = eval(model)
        model.save(f'{model.model_name}_{epoch}_{score}.pth')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseResNet('model50.pth').to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    data_dir = '../train'
    siamese_dataset = SiameseDataset(data_dir, transform=transform)
    train_loader = DataLoader(siamese_dataset, batch_size=32, shuffle=True, 
                              num_workers=6, pin_memory=True)
    # 开始训练
    train(model, train_loader, optimizer, num_epochs=1)
    
