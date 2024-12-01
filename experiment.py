import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import os
import copy
import matplotlib.pyplot as plt
import torch.nn.functional as F
from itertools import product

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 결과를 저장할 리스트 초기화
results_store = []

# 하이퍼파라미터 설정
batch_size = 64
learning_rate = 0.001
num_epochs = 5
temperature = 2.0  # Knowledge Distillation에서 사용되는 온도 매개변수
alpha = 0.7        # Distillation Loss와 Classification Loss의 가중치 조절
lambda_ewc = 1000  # EWC Regularization 강도
noise_dim = 100    # Generator의 노이즈 차원

# 데이터 변환 정의
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# MNIST 데이터셋 로드
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

task_classes = [
    [0, 1, 2, 3, 4, 5],    # Task 1 (6 classes)
    [6, 7, 8, 9]           # Task 2 (4 classes)
]

# 태스크별 데이터셋 생성 함수 수정
def get_task_dataset(dataset, classes):
    indices = [i for i, (_, label) in enumerate(dataset) if label in classes]
    subset = [(dataset[i][0], dataset[i][1]) for i in indices]
    images, labels = zip(*subset)
    dataset = TensorDataset(torch.stack(images), torch.tensor(labels))
    return dataset

# 태스크별 데이터셋 리스트 생성
task_train_datasets = [get_task_dataset(train_dataset, classes) for classes in task_classes]
task_test_datasets = [get_task_dataset(test_dataset, classes) for classes in task_classes]

# Teacher 모델 정의
class TeacherModel(nn.Module):
    def __init__(self, num_classes=10):
        super(TeacherModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(64 * 7 * 7, num_classes)

    def forward(self, x):
        feature_maps = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                feature_maps.append(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits, feature_maps

# Student 모델 정의
class StudentModel(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Layer 1: 1 -> 32
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Layer 2: 32 -> 64
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(64 * 7 * 7, num_classes)  # Linear 레이어 입력 64

    def forward(self, x):
        feature_maps = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                feature_maps.append(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
                
        return logits, feature_maps  # 로짓과 Feature Map 반환

# 수정된 Generator 모델 정의
class Generator(nn.Module):
    def __init__(self, noise_dim=100, img_channels=1, feature_maps=64):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # 입력: (batch_size, noise_dim, 1, 1)
            nn.ConvTranspose2d(noise_dim, feature_maps * 4, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # (batch_size, feature_maps*4, 7, 7)
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # (batch_size, feature_maps*2, 14, 14)
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # (batch_size, feature_maps, 28, 28)
            nn.ConvTranspose2d(feature_maps, img_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()  # 출력: (-1, 1, 28, 28)
        )
    
    def forward(self, x):
        return self.net(x)

# Knowledge Distillation을 위한 Distillation Loss 정의
def distillation_loss(student_logits, teacher_logits, temperature):
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    loss = nn.KLDivLoss(reduction='batchmean')(soft_student, soft_teacher) * (temperature ** 2)
    return loss

def attention_transfer_loss(student_features, teacher_features, cam_resolution=7, normalize=True):
    """
    Class Attention Transfer (CAT) Loss 계산 함수.

    Args:
        student_features (torch.Tensor): 학생 모델의 Feature Map (B, C, H, W)
        teacher_features (torch.Tensor): 교사 모델의 Feature Map (B, C, H, W)
        cam_resolution (int): CAM의 해상도 (기본값: 7)
        normalize (bool): CAM 정규화 여부 (기본값: True)

    Returns:
        torch.Tensor: Attention Transfer 손실 값
    """
    # Feature Map의 절대값 제곱하여 Attention Map 생성
    student_att = torch.sum(student_features ** 2, dim=1, keepdim=True)  # (B, 1, H, W)
    teacher_att = torch.sum(teacher_features ** 2, dim=1, keepdim=True)  # (B, 1, H, W)

    # CAM 해상도 조정
    student_att = F.adaptive_avg_pool2d(student_att, (cam_resolution, cam_resolution))
    teacher_att = F.adaptive_avg_pool2d(teacher_att, (cam_resolution, cam_resolution))

    # 정규화 옵션 적용
    if normalize:
        student_att = F.normalize(student_att, p=2, dim=(2, 3))
        teacher_att = F.normalize(teacher_att, p=2, dim=(2, 3))

    # MSE Loss 계산
    loss = F.mse_loss(student_att, teacher_att)
    return loss



class SI:
    def __init__(self, model, c=0.1, epsilon=0.1):
        self.model = model
        self.c = c
        self.epsilon = epsilon
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.omega = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        self.prev_params = {}
        self.device = next(model.parameters()).device

    def begin_task(self):
        self.prev_params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        self.small_omega = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}

    def update_weights(self):
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                delta = p.detach() - self.prev_params[n]
                self.small_omega[n] += -p.grad.detach() * delta

    def end_task(self):
        for n in self.params:
            delta = self.params[n] - self.prev_params[n]
            self.omega[n] += self.small_omega[n] / (delta ** 2 + self.epsilon)
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}

    def penalty(self):
        loss = 0
        for n, p in self.model.named_parameters():
            loss += (self.omega[n] * (p - self.params[n]) ** 2).sum()
        return self.c * loss

# EWC (Elastic Weight Consolidation)
class EWC:
    def __init__(self, model, dataset, lambda_=1000):
        self.params = []
        self.fishers = []
        self.lambda_ = lambda_
        self._add_model(model, dataset)

    def _add_model(self, model, dataset):
        params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        fisher = self._compute_fisher(model, dataset)
        self.params.append(params)
        self.fishers.append(fisher)
    
    def _compute_fisher(self, model, dataset, sample_size=200):
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        model.eval()
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        for i, (images, labels) in enumerate(loader):
            if i >= sample_size:
                break
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.clone().detach() ** 2
        # 평균 Fisher 정보
        for n in fisher:
            fisher[n] = fisher[n] / sample_size
        return fisher

    def penalty(self, model):
        loss = 0
        for old_params, old_fisher in zip(self.params, self.fishers):
            for n, p in model.named_parameters():
                if p.requires_grad:
                    if n not in old_fisher:
                        print(f"[EWC] Parameter '{n}' not found in Fisher information.")
                        continue
                    if p.shape != old_params[n].shape:
                        print(f"[EWC] Shape mismatch for parameter '{n}': current {p.shape}, stored {old_params[n].shape}")
                        continue
                    if old_fisher[n].shape != p.shape:
                        print(f"[EWC] Fisher shape mismatch for parameter '{n}': fisher {old_fisher[n].shape}, p {p.shape}")
                        continue
                    try:
                        loss += (old_fisher[n] * (p - old_params[n]) ** 2).sum()
                    except RuntimeError as e:
                        print(f"[EWC] Error processing parameter '{n}': {e}")
        return self.lambda_ * loss

def calculate_cam(feature_map, class_weights):
    """
    Class Activation Mapping (CAM) 계산.
    feature_map: (B, C, H, W) 형태
    class_weights: (C, ) 형태로 각 클래스의 가중치
    """
    return torch.einsum('bchw,c->bhw', feature_map, class_weights)

def apply_cam_options(teacher_features, logits_teacher, config):
    """
    CAM 전이 관련 옵션:
    - 바이너리화 (IF_BINARIZE)
    - 특정 클래스 CAM만 전이 (IF_OnlyTransferPartialCAMs)
    """
    if config['IF_BINARIZE']:
        # CAM 바이너리화
        n, c, h, w = teacher_features.shape
        threshold = torch.norm(teacher_features, dim=(2, 3), p=1, keepdim=True) / (h * w)
        teacher_features = (teacher_features > threshold).float()

    if config['IF_OnlyTransferPartialCAMs']:
        # 특정 클래스의 CAM만 전이
        n, c, h, w = teacher_features.shape
        top_classes = logits_teacher.topk(config['CAMs_Nums'], dim=1).indices  # 상위 클래스 선택
        mask = torch.zeros_like(teacher_features, dtype=torch.bool)
        for i in range(n):
            mask[i, top_classes[i]] = True
        teacher_features = teacher_features * mask.float()

    return teacher_features

def cat_loss(student_features, teacher_features, class_weights, cam_resolution, normalize=True):
    """
    CAT Loss 계산: Student와 Teacher의 CAM 비교.
    """
    student_cam = calculate_cam(student_features, class_weights)
    teacher_cam = calculate_cam(teacher_features, class_weights)
    
    # CAM 크기를 고정된 해상도로 조정
    student_cam = F.interpolate(student_cam.unsqueeze(1), size=(cam_resolution, cam_resolution), mode='bilinear').squeeze(1)
    teacher_cam = F.interpolate(teacher_cam.unsqueeze(1), size=(cam_resolution, cam_resolution), mode='bilinear').squeeze(1)
    
    # 정규화 옵션
    if normalize:
        student_cam = F.normalize(student_cam, dim=(1, 2))
        teacher_cam = F.normalize(teacher_cam, dim=(1, 2))
    
    # MSE Loss
    return F.mse_loss(student_cam, teacher_cam)

# Exemplar Replay를 위한 Exemplar 저장 클래스
class ExemplarStore:
    def __init__(self, max_size=100):
        self.exemplars = []
        self.max_size = max_size

    def add_exemplars(self, images, labels):
        for img, lbl in zip(images, labels):
            if len(self.exemplars) < self.max_size:
                self.exemplars.append((img.cpu().clone(), lbl.cpu().clone()))  # 저장 시 CPU로 이동
            else:
                # 무작위로 기존 데이터를 교체
                idx = np.random.randint(0, len(self.exemplars))
                self.exemplars[idx] = (img.cpu().clone(), lbl.cpu().clone())

    def get_replay_dataset(self):
        if not self.exemplars:
            return None
        images, labels = zip(*self.exemplars)
        images = torch.stack(images).to(device)  # 로드 시 GPU로 이동
        labels = torch.stack(labels).to(device)  # 로드 시 GPU로 이동
        return TensorDataset(images, labels)

# Pseudo Data Generation을 위한 함수 수정
def generate_pseudo_data(generator, num_samples=64, noise_dim=100):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, noise_dim, 1, 1).to(device)
        pseudo_data = generator(noise)  # Generator를 사용하여 이미지 생성
    return pseudo_data

# Data-Free KD 구현 예시
# Data-Free KD 구현
def data_free_kd(student_model, teacher_model, generator, num_samples=64, noise_dim=100, temperature=2.0, alpha=0.7):
    """
    Data-Free Knowledge Distillation (KD) 구현 함수.
    
    Args:
        student_model (nn.Module): Student 모델
        teacher_model (nn.Module): Teacher 모델
        generator (nn.Module): Pseudo Data Generator
        num_samples (int): 생성할 Pseudo 데이터 샘플 수
        noise_dim (int): Generator의 노이즈 입력 차원
        temperature (float): Distillation Loss 계산 시 온도 매개변수
        alpha (float): Distillation Loss와 Attention Transfer Loss의 가중치

    Returns:
        total_loss (torch.Tensor): Distillation Loss와 CAT Loss의 가중합
    """
    # Pseudo Data 생성
    pseudo_data = generate_pseudo_data(generator, num_samples=num_samples, noise_dim=noise_dim).to(device)
    
    # Teacher 모델의 출력 계산
    teacher_model.eval()  # Teacher 모델 평가 모드
    with torch.no_grad():
        teacher_outputs, teacher_features = teacher_model(pseudo_data)

    # Student 모델의 출력 계산
    student_outputs, student_features = student_model(pseudo_data)

    # Distillation Loss 계산
    distill_loss = distillation_loss(student_outputs, teacher_outputs, temperature)

    # CAT Loss 계산 (마지막 Feature Map 기준)
    cat_loss = attention_transfer_loss(student_features[-1], teacher_features[-1])

    # 최종 Loss 계산 (Distillation Loss와 CAT Loss의 가중합)
    total_loss = alpha * (distill_loss + cat_loss)

    return total_loss

def save_results(results_store, results_dir="experiment_results"):
    os.makedirs(results_dir, exist_ok=True)
    df_results_store = pd.DataFrame(results_store)
    results_store_file_path = os.path.join(results_dir, "results_store.csv")
    
    # CSV 파일을 쓰고 즉시 플러시
    with open(results_store_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        df_results_store.to_csv(csvfile, index=False)
        csvfile.flush()
        os.fsync(csvfile.fileno())  # 파일 디스크립터를 통해 강제로 디스크에 기록

    print(f"All results saved to '{results_store_file_path}'")
    return df_results_store

def visualize_results(df):
    plt.figure(figsize=(12, 6))
    for i, task_acc in enumerate(df['task_accuracies']):
        task_acc_list = eval(task_acc)  # 문자열을 리스트로 변환
        plt.plot(range(1, len(task_acc_list) + 1), task_acc_list, label=f'Experiment {i + 1}')
    plt.xlabel('Task ID')
    plt.ylabel('Accuracy (%)')
    plt.title('Task-wise Accuracy across Experiments')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Functional Regularization Loss 정의
def functional_regularization_loss(student_outputs, teacher_outputs):
    loss = nn.MSELoss()(student_outputs, teacher_outputs)
    return loss  # 전체 로짓 사용

# 수정된 Generator 학습 함수 정의
def train_generator(generator, teacher_model, task_train_dataset, num_epochs=5, noise_dim=100, generator_lr=1e-4):
    generator.train()
    generator.to(device)  # Generator를 GPU로 이동
    teacher_model.to(device)  # Teacher 모델도 GPU로 이동
    optimizer_g = optim.Adam(generator.parameters(), lr=generator_lr)  # 전달받은 generator_lr 사용
    criterion_g = nn.CrossEntropyLoss()

    train_loader = DataLoader(task_train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0
        for images, _ in train_loader:
            images = images.to(device)  # 데이터를 GPU로 이동
            batch_size_curr = images.size(0)
            noise = torch.randn(batch_size_curr, noise_dim, 1, 1).to(device)
            fake_images = generator(noise)

            teacher_logits, _ = teacher_model(fake_images)  # Teacher도 GPU에서 연산

            target_classes = torch.argmax(F.softmax(teacher_logits, dim=1), dim=1).to(device)
            loss = criterion_g(teacher_logits, target_classes) + nn.MSELoss()(fake_images, images)

            optimizer_g.zero_grad()
            loss.backward()
            optimizer_g.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Generator Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
    generator.eval()


def run_experiment(config, experiment_id, results_store):
    print(f"\n=== Running Experiment {experiment_id + 1}: Experiment {experiment_id + 1} ===")
    print(f"Configuration: {config}")

    # Student 모델 초기화 및 디바이스로 이동
    student_model = StudentModel(num_classes=10).to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
    classification_criterion = nn.CrossEntropyLoss()

    # Regularization 기법 초기화
    ewc = None  # EWC를 함수 시작 부분에서 None으로 초기화

    # Exemplar Replay 초기화
    # Exemplar Replay 초기화
    exemplar_store = ExemplarStore(max_size=config['exemplar_store_size']) if config['replay_method'] == 'exemplar' else None

    # Generator 및 previous_generators 초기화
    generator = None
    previous_generators = []

    if config.get('use_generator', False) or config.get('replay_method') == 'generative':
        generator = Generator(noise_dim=noise_dim).to(device)

    # 이전 Teacher 모델들을 저장할 리스트를 초기화합니다.
    previous_teacher_models = []

    # SI 초기화
    if config['reg_method'] == 'SI':
        si = SI(student_model)

    # 포겟팅 측정을 위한 리스트 초기화
    forgetting_metrics = []

    # 각 태스크별 정확도를 저장하기 위한 리스트 초기화
    task_accuracies = []

    # 태스크 루프 시작
    for task_idx, (task_train_dataset, task_test_dataset) in enumerate(zip(task_train_datasets, task_test_datasets)):
        print(f"\n[Experiment {experiment_id + 1}] Processing Task {task_idx + 1}: Classes {task_classes[task_idx]}")

        train_loader = DataLoader(task_train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(task_test_dataset, batch_size=batch_size, shuffle=False)

        # 각 태스크별로 EWC 초기화
        if config['reg_method'] == 'EWC':
            ewc = EWC(student_model, train_loader.dataset, lambda_=config['lambda_ewc'])
            
        # SI 시작
        if config['reg_method'] == 'SI':
            si.begin_task()

        # Teacher 모델 학습
        teacher_models = [
            TeacherModel(num_classes=10).to(device)
            for _ in range(config['num_teachers'])
        ]

        for i, teacher_model in enumerate(teacher_models):
            optimizer_teacher = optim.Adam(teacher_model.parameters(), lr=learning_rate)
            criterion_teacher = nn.CrossEntropyLoss()

            print(f"Training Teacher Model {i + 1} for Task {task_idx + 1}: Classes {task_classes[task_idx]}")
            for epoch in range(num_epochs):
                teacher_model.train()
                total_loss = 0
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer_teacher.zero_grad()
                    outputs, _ = teacher_model(images)

                    loss = criterion_teacher(outputs, labels)
                    loss.backward()
                    optimizer_teacher.step()
                    total_loss += loss.item()
                avg_loss = total_loss / len(train_loader)
                print(f"Teacher Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
            teacher_model.eval()

        # Generator 학습
        if generator is not None:
            print(f"Training Generator for Task {task_idx + 1}")
            train_generator(generator, teacher_model, task_train_dataset,
                num_epochs=config['generator_num_epochs'], noise_dim=noise_dim, generator_lr=config['generator_lr'])
            previous_generators.append(copy.deepcopy(generator))

        # Exemplar Replay 추가
        if exemplar_store is not None:
            for images, labels in train_loader:
                exemplar_store.add_exemplars(images, labels)

        # Student 모델 학습 루프 내에서
        for epoch in range(num_epochs):
            student_model.train()
            total_loss, total_correct, total_samples = 0, 0, 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                # Student 모델의 출력 및 Feature Map 얻기
                student_outputs, student_features = student_model(images)

                # 현재 태스크의 Teacher 모델의 출력 및 Feature Map 얻기
                teacher_outputs, teacher_features = teacher_model(images)

                # Classification Loss 계산
                cls_loss = classification_criterion(student_outputs, labels)

                # 현재 태스크의 Distillation Loss 계산
                distill_loss_current = 0
                if config['kd_method'] == 'vanilla' or config['kd_method'] == 'logit':
                    distill_loss_current = distillation_loss(student_outputs, teacher_outputs, config['temperature'])
                elif config['kd_method'] == 'attention':
                    distill_loss_current = attention_distillation_loss(student_features, teacher_features)
                elif config['kd_method'] == 'feature':
                    distill_loss_current = feature_distillation_loss(student_features, teacher_features)
                elif config['kd_method'] == 'cat':
                    distill_loss_current = attention_transfer_loss(student_features[-1], teacher_features[-1])
                else:
                    raise ValueError(f"Unknown kd_method: {config['kd_method']}")

                # 이전 태스크에 대한 Distillation Loss 계산
                distill_loss_prev = 0

                for prev_idx, prev_teacher_model in enumerate(previous_teacher_models):
                    prev_teacher_model.eval()
                    with torch.no_grad():
                        prev_teacher_outputs, prev_teacher_features = prev_teacher_model(images)

                    if config['kd_method'] == 'vanilla' or config['kd_method'] == 'logit':
                        distill_loss_prev += distillation_loss(student_outputs, prev_teacher_outputs, config['temperature'])
                    elif config['kd_method'] == 'attention':
                        distill_loss_prev += attention_distillation_loss(student_features, prev_teacher_features)
                    elif config['kd_method'] == 'feature':
                        distill_loss_prev += feature_distillation_loss(student_features, prev_teacher_features)
                    elif config['kd_method'] == 'cat':
                        distill_loss_prev += attention_transfer_loss(student_features[-1], prev_teacher_features[-1])

                # Replay Method 적용
                if config['replay_method'] == 'exemplar' and exemplar_store is not None:
                    replay_dataset = exemplar_store.get_replay_dataset()
                    if replay_dataset is not None:
                        replay_loader = DataLoader(replay_dataset, batch_size=batch_size, shuffle=True)
                        for replay_images, replay_labels in replay_loader:
                            replay_images, replay_labels = replay_images.to(device), replay_labels.to(device)
                            student_replay_outputs, student_replay_features = student_model(replay_images)
                            with torch.no_grad():
                                teacher_replay_outputs, teacher_replay_features = teacher_model(replay_images)

                            if config['kd_method'] == 'vanilla' or config['kd_method'] == 'logit':
                                distill_loss_prev += distillation_loss(student_replay_outputs, teacher_replay_outputs, config['temperature'])
                            elif config['kd_method'] == 'attention':
                                distill_loss_prev += attention_distillation_loss(student_replay_features, teacher_replay_features)
                            elif config['kd_method'] == 'feature':
                                distill_loss_prev += feature_distillation_loss(student_replay_features, teacher_replay_features)
                            elif config['kd_method'] == 'cat':
                                distill_loss_prev += attention_transfer_loss(student_replay_features[-1], teacher_replay_features[-1])

                elif config['replay_method'] == 'generative' and previous_generators:
                    for prev_generator, prev_teacher_model in zip(previous_generators, previous_teacher_models):
                        pseudo_data = generate_pseudo_data(prev_generator, num_samples=batch_size, noise_dim=noise_dim)
                        pseudo_data = pseudo_data.to(device)
                        student_pseudo_outputs, student_pseudo_features = student_model(pseudo_data)
                        with torch.no_grad():
                            prev_teacher_outputs, prev_teacher_features = prev_teacher_model(pseudo_data)

                        if config['kd_method'] == 'vanilla' or config['kd_method'] == 'logit':
                            distill_loss_prev += distillation_loss(student_pseudo_outputs, prev_teacher_outputs, config['temperature'])
                        elif config['kd_method'] == 'attention':
                            distill_loss_prev += attention_distillation_loss(student_pseudo_features, prev_teacher_features)
                        elif config['kd_method'] == 'feature':
                            distill_loss_prev += feature_distillation_loss(student_pseudo_features, prev_teacher_features)
                        elif config['kd_method'] == 'cat':
                            distill_loss_prev += attention_transfer_loss(student_pseudo_features[-1], prev_teacher_features[-1])

                # 총 Distillation Loss 계산
                total_distill_loss = distill_loss_current + distill_loss_prev

                # Data-Free KD 적용
                data_free_loss = 0
                if config['data_free_kd_option'] == 'pseudo' and generator is not None:
                    data_free_loss = data_free_kd(student_model, teacher_model, generator,
                                                  num_samples=batch_size, noise_dim=noise_dim,
                                                  temperature=config['temperature'], alpha=alpha)

                # 최종 Loss 계산
                loss = alpha * (total_distill_loss + data_free_loss) + (1 - alpha) * cls_loss

                # Regularization 기법(EWC 등) 적용
                if ewc is not None:
                    ewc_loss = ewc.penalty(student_model)
                    loss += ewc_loss

                # SI 패널티 추가
                if config['reg_method'] == 'SI':
                    loss += si.penalty()

                loss.backward()

                # SI 업데이트
                if config['reg_method'] == 'SI':
                    si.update_weights()

                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(student_outputs, dim=1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

            train_acc = 100 * total_correct / total_samples
            avg_loss = total_loss / len(train_loader)

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # SI 종료
        if config['reg_method'] == 'SI':
            si.end_task()

        # 테스트 정확도 측정
        student_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs, _ = student_model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_acc = 100 * correct / total
        print(f"Test Accuracy: {test_acc:.2f}%")

        # 각 태스크의 정확도를 저장
        task_accuracies.append({
            "task_id": task_idx + 1,
            "accuracy": test_acc
        })

        # Forgetting 측정
        if task_idx > 0:
            forgetting = []
            for prev_idx, prev_dataset in enumerate(task_test_datasets[:task_idx]):
                prev_loader = DataLoader(prev_dataset, batch_size=batch_size, shuffle=False)
                prev_correct, prev_total = 0, 0
                with torch.no_grad():
                    for prev_images, prev_labels in prev_loader:
                        prev_images, prev_labels = prev_images.to(device), prev_labels.to(device)
                        outputs, _ = student_model(prev_images)
                        _, predicted = torch.max(outputs, 1)
                        prev_total += prev_labels.size(0)
                        prev_correct += (predicted == prev_labels).sum().item()
                prev_acc = 100 * prev_correct / prev_total
                print(f"Accuracy on Task {prev_idx + 1}: {prev_acc:.2f}%")
                previous_best_acc = max([acc["accuracy"] for acc in task_accuracies if acc["task_id"] == prev_idx + 1], default=0)
                forgetting.append(max(previous_best_acc - prev_acc, 0))
            avg_forgetting = np.mean(forgetting)
            forgetting_metrics.append(float(avg_forgetting))
            print(f"Average Forgetting after Task {task_idx + 1}: {avg_forgetting:.2f}%")
        else:
            avg_forgetting = 0

        # 이전 Teacher 모델 저장
        previous_teacher_models.append(copy.deepcopy(teacher_model))

    # 전체 평균 포겟팅 계산
    total_forgetting = np.mean(forgetting_metrics) if forgetting_metrics else 0

    # 결과 저장
    results_store.append({
        "description": config["description"],
        "test_acc": test_acc,
        "total_forgetting": float(total_forgetting),  # float으로 변환
        "forgetting_per_task": [float(f) for f in forgetting_metrics],  # float 변환
        "task_accuracies": task_accuracies,  # 태스크별 정확도
        "softmax": config["softmax"],  # softmax 추가
    })
    print(f"[Experiment {experiment_id + 1}] Finished.")
    
    # 각 실험이 끝날 때마다 결과를 저장하고 플러시
    df_results_store = pd.DataFrame(results_store)
    results_store_file_path = os.path.join("experiment_results", "results_store.csv")
    df_results_store.to_csv(results_store_file_path, index=False)
    print(f"Results saved up to Experiment {experiment_id + 1}")

# Attention Distillation Loss 정의
def attention_distillation_loss(student_features, teacher_features):
    loss = 0
    for sf, tf in zip(student_features, teacher_features):
        # Attention Map 계산
        student_att = torch.sum(sf ** 2, dim=1, keepdim=True)
        teacher_att = torch.sum(tf ** 2, dim=1, keepdim=True)
        # 정규화
        student_att = F.normalize(student_att.view(student_att.size(0), -1), p=2, dim=1)
        teacher_att = F.normalize(teacher_att.view(teacher_att.size(0), -1), p=2, dim=1)
        # MSE Loss 계산
        loss += F.mse_loss(student_att, teacher_att)
    return loss

### ADDITION ### Feature Distillation Loss 정의
def feature_distillation_loss(student_features, teacher_features):
    loss = 0
    for sf, tf in zip(student_features, teacher_features):
        # Feature Map 사이의 MSE Loss 계산
        loss += F.mse_loss(sf, tf)
    return loss

num_teachers_list = [2]
temperature_list = [1.0]
softmax_list = [False] # True, False
kd_method_list = ['logit'] # 'cat', 'vanilla', 'logit', 'attention', 'feature'
reg_method_list = ['SI'] # None, 'EWC', 'SI'
replay_method_list = [None] # None, 'exemplar', 'generative'
data_free_kd_list = [None] # None, 'pseudo'
use_generator_list = [False] # True, False
onlyCAT_list = [False] # True, False
IF_BINARIZE_list = [False] # True, False
IF_OnlyTransferPartialCAMs_list = [False] # True, False
CAMs_Nums_list = [1] # 1, 3, 5
generator_lr_list = [1e-4]


# 새로 추가된 실험 변수
lambda_ewc_list = [500, 1000, 5000, 10000]  # EWC 강도
exemplar_store_size_list = [50, 100, 500, 1000]  # Exemplar Replay 크기
generator_num_epochs_list = [5, 10, 20]  # Generator 학습 Epoch

experiment_configs = []
experiment_id = 0

def create_description(config):
    return (
        f"Experiment with num_teachers={config['num_teachers']}, "
        f"temperature={config['temperature']}, kd_method={config['kd_method']}, "
        f"reg_method={config['reg_method']}, replay_method={config['replay_method']}, "
        f"data_free_kd={config['data_free_kd_option']}, "
        f"use_generator={config['use_generator']}, "
        f"onlyCAT={config['onlyCAT']}, IF_BINARIZE={config['IF_BINARIZE']}, "
        f"IF_OnlyTransferPartialCAMs={config['IF_OnlyTransferPartialCAMs']}, "
        f"CAMs_Nums={config['CAMs_Nums']}, generator_lr={config['generator_lr']}, "
        f"lambda_ewc={config['lambda_ewc']}, exemplar_store_size={config['exemplar_store_size']}, "
        f"generator_num_epochs={config['generator_num_epochs']}, softmax={config['softmax']}"
    )

# Config 조합 생성
for num_teachers, temperature, kd_method, reg_method, replay_method, data_free_kd_option, use_generator, onlyCAT, IF_BINARIZE, IF_OnlyTransferPartialCAMs, CAMs_Nums, generator_lr, lambda_ewc, exemplar_size, generator_num_epochs, softmax in product(
    num_teachers_list,
    temperature_list,
    kd_method_list,
    reg_method_list,
    replay_method_list,
    data_free_kd_list,
    use_generator_list,
    onlyCAT_list,
    IF_BINARIZE_list,
    IF_OnlyTransferPartialCAMs_list,
    CAMs_Nums_list,
    generator_lr_list,
    lambda_ewc_list,
    exemplar_store_size_list,
    generator_num_epochs_list,
    softmax_list,
):
    # 반복문 내부 로직
    config = {
        'num_teachers': num_teachers,
        'temperature': temperature,
        'kd_method': kd_method,
        'reg_method': reg_method,
        'replay_method': replay_method,
        'data_free_kd_option': data_free_kd_option,
        'use_generator': use_generator,
        'onlyCAT': onlyCAT,
        'IF_BINARIZE': IF_BINARIZE,
        'IF_OnlyTransferPartialCAMs': IF_OnlyTransferPartialCAMs,
        'CAMs_Nums': CAMs_Nums,
        'generator_lr': generator_lr,
        'lambda_ewc': lambda_ewc,
        'exemplar_store_size': exemplar_size,
        'generator_num_epochs': generator_num_epochs,
        'softmax': softmax,
    }

    # Description 추가
    config['experiment_description'] = create_description(config)

    # 설정 추가
    experiment_configs.append(config)
    experiment_id += 1

print(f"총 실험 수: {len(experiment_configs)}")


# 실험 실행
for experiment_id, config in enumerate(experiment_configs):
    try:
        if 'description' not in config:  # description 키가 없으면 기본값 추가
            config['description'] = create_description(config)

        run_experiment(config, experiment_id, results_store)
    except Exception as e:
        print(f"Experiment failed for config: {config.get('description', 'No description provided')}\nError: {e}")
        results_store.append({
            "description": config.get('description', 'No description available'),
            "test_acc": None,
            "total_forgetting": None,
            "forgetting_per_task": None,
            "task_accuracies": None,
            "error": str(e),
        })
        save_results(results_store, results_dir="experiment_results")
        with open("experiment_results/error_log.txt", "a") as f:
            f.write(f"Experiment {experiment_id + 1}: {config.get('description', 'No description provided')}\n")
            f.write(f"Error: {str(e)}\n\n")

# 실험 결과를 저장할 디렉토리 설정
project_dir = os.getcwd()  # 현재 작업 디렉토리로 설정
results_dir = os.path.join(project_dir, "experiment_results")
os.makedirs(results_dir, exist_ok=True)

# 실험 결과 저장
df_results_store = save_results(results_store, results_dir="experiment_results")

# 시각화 호출
# visualize_results(df_results_store)
# 시각화는 사용하지않고, 데이터를 csv파일로 받아 데이터분석을 실시하였음
