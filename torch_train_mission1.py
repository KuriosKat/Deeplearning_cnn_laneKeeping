import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

batch_size = 16
num_epochs = 70
learning_rate = 0.0001

# 이미지 데이터셋 경로 설정
data_path = "Augment_in_busan"  # data 폴더 안에 각 클래스 폴더 (0, 1, 2)가 있어야 함

# 데이터 전처리 및 DataLoader 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = ImageFolder(root=data_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 딥러닝 모델 정의 (예시로 간단한 CNN 모델을 사용)
class LaneClassifier(nn.Module):
    '''
    def __init__(self):
        super(LaneClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 32 * 56 * 56)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    '''
    def __init__(self):
        super(LaneClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 5)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x)) #frelu, is good
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 128 * 28 * 28)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 모델 생성 및 손실함수, 최적화 알고리즘 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = LaneClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 이전 학습코드
'''
for epoch in range(num_epochs):
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# 모델 저장
torch.save(model.state_dict(), "untrained_dataset_mission1_0813_0208_YH_MCKcontrol.pth")
'''
save_interval = 3
for epoch in range(num_epochs):
    total_loss = 0.0
    tp, fp, fn = 0, 0, 0  # True Positive, False Positive, False Negative 초기화
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        
        # True Positive, False Positive, False Negative 계산
        for pred, label in zip(preds, labels):
            if pred == label:
                tp += 1
            else:
                fp += (pred == 1)  # 예측이 양성(1)인데 실제는 음성(0)인 경우
                fn += (pred == 0)  # 예측이 음성(0)인데 실제는 양성(1)인 경우
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    precision = tp / (tp + fp + 1e-9)  # 분모가 0이 되지 않도록 더해줌
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)  # F1 score 계산
    print(f'F1 Score: {f1:.4f}')

    if (epoch + 1) % save_interval == 0:
        model_filename = f"lane_classifier_model_class5_epoch{epoch+1}_0817.pth"
        torch.save(model.state_dict(), model_filename)
        print(f"Saved model weights at epoch {epoch+1}")

# 모델 저장
torch.save(model.state_dict(), "lane_classifier_model_class5_epoch70_0817.pth")


