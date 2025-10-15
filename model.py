import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import polars as pl
import datetime as dt
import ast
from PIL import Image
from torchvision import transforms

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def convert_image_to_list(image_path):
    image = Image.open(f"images/{image_path}").convert("RGB")
    transform = transforms.ToTensor()
    tensor = transform(image)
    return tensor


def parse_detection(detection_str):
    """
    detection 문자열을 파싱하여 텐서로 변환합니다.
    detection 형식: "class_id,center_x,center_y,width,height" 또는 빈 문자열
    여러 detection이 있을 경우 세미콜론(;)으로 구분됩니다.

    Args:
        detection_str: detection 정보 문자열

    Returns:
        첫 번째 detection의 [center_x, center_y, width, height] (4차원)
        detection이 없으면 [0.0, 0.0, 0.0, 0.0] 반환
    """
    if not detection_str or detection_str == '':
        return [0.0, 0.0, 0.0, 0.0]

    # 세미콜론으로 구분된 detection 중 첫 번째만 사용 (컵이 여러 개일 경우)
    first_detection = detection_str.split(';')[0]
    values = first_detection.split(',')

    if len(values) >= 5:
        # class_id는 제외하고 center_x, center_y, width, height만 반환
        return [float(values[1]), float(values[2]), float(values[3]), float(values[4])]

    return [0.0, 0.0, 0.0, 0.0]


# --- 1. 가상 데이터셋 클래스 정의 ---
class RobotGraspingDataset(Dataset):
    def __init__(self, csv_file_path=None):
        self.df = pl.read_csv(csv_file_path)

        # 1. 입력 이미지 (RGB, 640X480)
        self.images = self.df.select(self.df["image_name"].map_elements(convert_image_to_list)).get_column(
            "image_name").to_list()
        print(f"Image shape: {self.images[0].shape}")

        # 2. 현재 로봇 자세 좌표 (x, y, z, rx, ry, rz) - 6차원
        self.df = self.df.with_columns(
            pl.col("robot_base_coordinate").map_elements(ast.literal_eval, return_dtype=pl.List(pl.Float64)))
        self.current_poses = torch.tensor(self.df["robot_base_coordinate"].to_list())

        # 3. 목표 좌표 (x, y, z, rx, ry, rz) - 출력(정답)
        self.df = self.df.with_columns(
            pl.col("robot_target_coordinate").map_elements(ast.literal_eval, return_dtype=pl.List(pl.Float64)))
        self.target_poses = torch.tensor(self.df["robot_target_coordinate"].to_list())

        # 4. Detection 정보 (center_x, center_y, width, height) - 4차원
        self.df = self.df.with_columns(
            pl.col("detection").map_elements(parse_detection, return_dtype=pl.List(pl.Float64))
        )
        self.detections = torch.tensor(self.df["detection"].to_list())
        print(f"Detection shape: {self.detections[0].shape}")
        print(f"Sample detection: {self.detections[0]}")

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        # 입력: 이미지, 현재 자세, detection 정보
        # 출력: 목표 자세
        return self.images[idx], self.current_poses[idx], self.detections[idx], self.target_poses[idx]


# 데이터셋 인스턴스화
train_dataset = RobotGraspingDataset(csv_file_path="result.csv")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
print(f"Dataset Size: {len(train_dataset)}")


# --- 2. 멀티모달 신경망 모델 정의 ---
class VisionPoseNet(nn.Module):
    def __init__(self):
        super(VisionPoseNet, self).__init__()

        # 1. 이미지 처리 모듈 (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16 x 240 x 320
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 x 120 x 160
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 x 60 x 80
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128 x 30 x 40
            nn.Flatten(),
            nn.Linear(128 * 30 * 40, 128)  # 153600 -> 128
        )

        # 2. 현재 자세 처리 모듈 (MLP)
        # 현재 자세 (x, y, z, rx, ry, rz) 6차원을 받아 32차원 특징 벡터 출력
        self.pose_mlp = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )

        # 3. Detection 처리 모듈 (MLP)
        # Detection (center_x, center_y, width, height) 4차원을 받아 16차원 특징 벡터 출력
        self.detection_mlp = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16)
        )

        # 4. 특징 융합 및 목표 자세 예측 모듈 (Fusion MLP)
        # CNN 특징(128) + Pose 특징(32) + Detection 특징(16) = 176차원 입력
        self.fusion_mlp = nn.Sequential(
            nn.Linear(128 + 32 + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 6)  # 최종 출력: 목표 자세 (x, y, z, rx, ry, rz) 6차원
        )

    def forward(self, image, current_pose, detection):
        # 1. 이미지 특징 추출
        visual_features = self.cnn(image)

        # 2. 현재 자세 특징 추출
        pose_features = self.pose_mlp(current_pose)

        # 3. Detection 특징 추출
        detection_features = self.detection_mlp(detection)

        # 4. 특징 융합 (Concatenation)
        fused_features = torch.cat((visual_features, pose_features, detection_features), dim=1)

        # 5. 목표 자세 예측
        target_pose_pred = self.fusion_mlp(fused_features)

        return target_pose_pred


# 모델 인스턴스화 및 장치로 이동
model = VisionPoseNet().to(device)
print("\nModel Architecture:\n", model)

# --- 3. 모델 학습 ---
criterion = nn.MSELoss()  # 평균 제곱 오차: 회귀 문제에 적합
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 80
print(f"\nStarting training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, current_poses, detections, target_poses in train_loader:
        # 데이터를 장치(GPU/CPU)로 이동
        images = images.to(device)
        current_poses = current_poses.to(device)
        detections = detections.to(device)
        target_poses = target_poses.to(device)

        # 그라디언트 초기화
        optimizer.zero_grad()
        # 순전파
        predicted_poses = model(images, current_poses, detections)

        # 손실 계산
        loss = criterion(predicted_poses, target_poses)

        # 역전파 및 가중치 업데이트
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.6f}")

print("Training finished!")

# --- 4. 추론 (Inference) 예시 ---
model.eval()  # 추론 모드로 전환

# 가상의 새로운 입력 데이터 생성
test_image_name = "test_1.jpg"
test_image = Image.open(f"test_images/{test_image_name}").convert("RGB")
test_transform = transforms.ToTensor()
test_tensor = test_transform(test_image)
new_image = test_tensor.unsqueeze(0).to(device)
new_current_pose = torch.tensor([-71.2, 26.8, 296.6, -141.61, 32.68, 152.02]).unsqueeze(0).to(device)
# 테스트용 detection 정보 (실제로는 객체 탐지 모델의 출력을 사용)
new_detection = torch.tensor([0.5478591918945312, 0.22486144304275513, 0.24161967635154724, 0.4224931299686432]).unsqueeze(0).to(device)  # center_x, center_y, width, height

with torch.no_grad():
    predicted_target_pose = model(new_image, new_current_pose, new_detection)

# 결과 출력
print("\n--- Inference Result ---")
print(f"Input Current Pose (x, y, z, rx, ry, rz): {new_current_pose.cpu().numpy()}")
print(f"Input Detection (center_x, center_y, width, height): {new_detection.cpu().numpy()}")
print(f"Predicted Target Pose (x, y, z, rx, ry, rz): {predicted_target_pose.cpu().numpy()}")