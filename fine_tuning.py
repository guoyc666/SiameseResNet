import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from modelS import SiameseResNet
from tqdm import tqdm
from test import eval
from make_test_data import generate_test_data

train_dir = '../data/train'  # 训练集路径
val_dir = '../data/valid'      # 验证集路径
class_num = 700
num_epochs = 10   # 根据数据集大小与任务需求调整
learning_rate = 5e-5
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据增强与预处理
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 训练函数
def train_one_epoch(model, dataloader, criterion, optimizer, description):
    # return 0, 0
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    device = next(model.parameters()).device

    for inputs, labels in tqdm(dataloader, desc=description):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播与优化
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy

# 验证函数
def validate_model(model, dataloader, criterion, description):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=description):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy


if __name__ == '__main__':
    if not os.path.exists('./fine_tuned'):
        os.makedirs('./fine_tuned')
    if not os.path.exists('./out'):
        os.makedirs('./out')
    generate_test_data(10)

    # 数据加载器
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    # val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
    #                         num_workers=4, pin_memory=True)

    # 加载预训练模型并修改输出层
    # model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, class_num)

    static_dict = torch.load('./fine_tuned/resnet50_2.pth', weights_only=True, map_location=device)
    model.load_state_dict(static_dict)
    
    model = model.to(device)

    # 设置优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.fc.parameters(), 'lr': learning_rate * 10},
        {'params': [param for name, param in model.named_parameters() if "fc" not in name],
        'lr': learning_rate}],
        weight_decay=1e-3
        )
    # 训练与验证流程
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, 
                                                     criterion, optimizer, 
                                                     f"[{epoch+1}/{num_epochs}]")
        # val_loss, val_accuracy = validate_model(model, val_loader, criterion, 
                                                # f"[{epoch+1}/{num_epochs}] Validating")
        
        save_path = f'./fine_tuned/resnet50_x_{epoch+3}.pth'
        torch.save(model.state_dict(), save_path)

        s = SiameseResNet(save_path).to(device)
        scores = eval(s)
        print(f"Score: {scores}\n")
        
        with open('./out/fine_tuned_resnet50.txt', 'a') as f:
            f.write(f"X Epoch {epoch+3}\n")
            f.write(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}\n")
            # f.write(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}\n")
            f.write(f"Score: {scores}\n")
            f.write("\n")


