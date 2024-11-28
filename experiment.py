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

# 태스크 분할: 5개의 태스크 (각 태스크마다 2개의 클래스)
task_classes = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9]
]

# 태스크별 데이터셋 생성 함수
def get_task_dataset(dataset, classes):
    indices = [i for i, (_, label) in enumerate(dataset) if label in classes]
    subset = Subset(dataset, indices)
    return subset

# 태스크별 데이터셋 리스트 생성
task_train_datasets = [get_task_dataset(train_dataset, classes) for classes in task_classes]
task_test_datasets = [get_task_dataset(test_dataset, classes) for classes in task_classes]

# Teacher 모델 정의
class TeacherModel(nn.Module):
    def __init__(self, num_classes):
        super(TeacherModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [32,1,3,3]
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [64,32,3,3]
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(64 * 7 * 7, num_classes)  # [num_classes, 64*7*7]

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits
    

# Student 모델 정의
class StudentModel(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # [16, 1, 3, 3]
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # [32, 16, 3, 3]
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(32 * 7 * 7, num_classes)  # [10, 32*7*7]

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits

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
    soft_student = nn.functional.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = nn.functional.softmax(teacher_logits / temperature, dim=1)
    loss = nn.KLDivLoss(reduction='batchmean')(soft_student, soft_teacher) * (temperature ** 2)
    return loss

# EWC (Elastic Weight Consolidation) 구현
class EWC:
    def __init__(self, model, dataset, lambda_=1000):
        self.models = []
        self.lambdas = []
        self.params = []
        self.fishers = []
        self.lambda_ = lambda_
        self._add_model(model, dataset)

    def _add_model(self, model, dataset):
        params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        fisher = self._compute_fisher(model, dataset)
        self.models.append(copy.deepcopy(model))
        self.params.append(params)
        self.fishers.append(fisher)

    def _compute_fisher(self, model, dataset):
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        model.eval()
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.clone().detach() ** 2
        # 평균 Fisher 정보
        for n in fisher:
            fisher[n] = fisher[n] / len(loader)
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

# Class Attention Transfer (CAT) Loss 정의
def attention_transfer_loss(student_features, teacher_features):
    # 각 Feature Map의 평균을 계산하여 Attention 맵으로 사용
    student_att = torch.mean(student_features, dim=1)
    teacher_att = torch.mean(teacher_features, dim=1)
    loss = nn.MSELoss()(student_att, teacher_att)
    return loss

# Exemplar Replay를 위한 Exemplar 저장 클래스
class ExemplarStore:
    def __init__(self, max_size=100):
        self.exemplars = []
        self.max_size = max_size

    def add_exemplars(self, images, labels):
        for img, lbl in zip(images, labels):
            if len(self.exemplars) < self.max_size:
                self.exemplars.append((img.clone(), lbl.clone()))
            else:
                break  # 최대 크기 도달 시 추가하지 않음

    def get_replay_dataset(self):
        if not self.exemplars:
            return None
        images, labels = zip(*self.exemplars)
        images = torch.stack(images)
        labels = torch.stack(labels)
        return TensorDataset(images, labels)

# Pseudo Data Generation을 위한 함수 수정
def generate_pseudo_data(generator, num_samples=64, noise_dim=100):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, noise_dim, 1, 1).to(device)
        pseudo_data = generator(noise)  # Generator를 사용하여 이미지 생성
    return pseudo_data

# Data-Free KD 구현 예시
def data_free_kd(student_model, teacher_model, generator, num_samples=64, noise_dim=100):
    # Pseudo 데이터를 생성
    pseudo_data = generate_pseudo_data(generator, num_samples=num_samples, noise_dim=noise_dim)
    teacher_outputs = teacher_model(pseudo_data)
    student_outputs = student_model(pseudo_data)
    loss = distillation_loss(student_outputs, teacher_outputs, temperature)
    return loss

# Functional Regularization Loss 정의
def functional_regularization_loss(student_outputs, teacher_outputs):
    loss = nn.MSELoss()(student_outputs, teacher_outputs)
    return loss

# 수정된 Generator 학습 함수 정의
def train_generator(generator, teacher_model, task_train_dataset, num_epochs=5, noise_dim=100):
    generator.train()
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
    criterion_g = nn.MSELoss()  # 단순 예시로 MSE Loss 사용
    
    train_loader = DataLoader(task_train_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for images, _ in train_loader:
            batch_size_curr = images.size(0)
            noise = torch.randn(batch_size_curr, noise_dim, 1, 1).to(device)
            fake_images = generator(noise)
            
            #print(f"Fake images shape: {fake_images.shape}")  # 디버깅 출력 추가
            
            teacher_outputs = teacher_model(fake_images)
            
            # 단순히 Teacher 모델이 출력하는 로짓을 1로 만들도록 Generator 학습
            # 실제로는 더 정교한 방법이 필요함
            target = torch.ones_like(teacher_outputs).to(device)
            loss = criterion_g(teacher_outputs, target)
            
            optimizer_g.zero_grad()
            loss.backward()
            optimizer_g.step()
            
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Generator Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
    generator.eval()

# 결과를 DataFrame으로 변환 및 저장
df_results_store = pd.DataFrame(results_store)
results_store_file_path = os.path.join("experiment_results", "results_store.csv")
df_results_store.to_csv(results_store_file_path, index=False)

print(f"All results saved to '{results_store_file_path}'")

# Teacher 모델 수를 동적으로 설정
def create_teacher_models(num_teachers, task_classes):
    teacher_models = []
    for i in range(num_teachers):
        classes_for_teacher = task_classes[i % len(task_classes)]  # 순환적으로 클래스 할당
        teacher_model = TeacherModel(num_classes=len(classes_for_teacher)).to(device)
        teacher_models.append((teacher_model, classes_for_teacher))
    return teacher_models

# Knowledge Distillation에서 Temperature Scaling 추가
def distillation_loss(student_logits, teacher_logits, temperature):
    soft_student = nn.functional.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = nn.functional.softmax(teacher_logits / temperature, dim=1)
    loss = nn.KLDivLoss(reduction='batchmean')(soft_student, soft_teacher) * (temperature ** 2)
    return loss

# Forgetting 지표 계산 함수
def calculate_forgetting(previous_accuracies, current_accuracies):
    forgetting = {}
    for task_idx in previous_accuracies:
        if task_idx in current_accuracies:
            forgetting[task_idx] = previous_accuracies[task_idx] - current_accuracies[task_idx]
    return forgetting


def run_experiment(config, experiment_id, results_store):
    print(f"\n=== Running Experiment {experiment_id + 1}: {config['description']} ===")
    print(f"Configuration: {config}")
    
    # Student 모델 초기화 및 디바이스로 이동
    student_model = StudentModel(num_classes=10).to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
    classification_criterion = nn.CrossEntropyLoss()
    
    # Regularization 기법 초기화 (실험 시작 시점에서는 None)
    ewc = None
    
    # Exemplar Replay 초기화
    exemplar_store = ExemplarStore(max_size=100) if config['replay_method'] == 'exemplar' else None
    
    # Generator 초기화
    generator = Generator(noise_dim=noise_dim).to(device)
    
    # 이전 Teacher 모델들을 저장하기 위한 리스트
    previous_teacher_models = []
    forgetting_metrics = []
    initial_accuracies = []

    for task_idx, (task_train_dataset, task_test_dataset) in enumerate(zip(task_train_datasets, task_test_datasets)):
        print(f'\n[Experiment {experiment_id + 1}] Processing Task {task_idx + 1}')
        
        train_loader = DataLoader(task_train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(task_test_dataset, batch_size=batch_size, shuffle=False)
        
        # Teacher 모델 학습
        teacher_model = TeacherModel(num_classes=len(task_classes[task_idx])).to(device)
        optimizer_teacher = optim.Adam(teacher_model.parameters(), lr=learning_rate)
        criterion_teacher = nn.CrossEntropyLoss()
        
        print(f"Training Teacher Model for Task {task_idx + 1}")
        for epoch in range(num_epochs):
            teacher_model.train()
            total_loss = 0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer_teacher.zero_grad()
                outputs = teacher_model(images)
                # 라벨을 0부터 시작하도록 변경
                adjusted_labels = labels - min(task_classes[task_idx])
                loss = criterion_teacher(outputs, adjusted_labels)
                loss.backward()
                optimizer_teacher.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            print(f"Teacher Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
        teacher_model.eval()
        
        # Generator 학습
        print(f"Training Generator for Task {task_idx + 1}")
        train_generator(generator, teacher_model, task_train_dataset, num_epochs=num_epochs, noise_dim=noise_dim)
        
        # Exemplar Replay에 데이터 추가
        if exemplar_store is not None:
            exemplar_store.add_exemplars(images, labels)
        
        # 이전 Teacher 모델들 업데이트
        previous_teacher_models.append(teacher_model)
        
        # Student 모델 학습
        for epoch in range(num_epochs):
            student_model.train()
            train_total_loss = 0
            train_total_correct = 0
            train_total_samples = 0
            
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                student_outputs = student_model(images)
                teacher_outputs = teacher_model(images)
                
                # Student 출력에서 현재 태스크의 클래스에 해당하는 로짓만 선택
                student_logits_for_task = student_outputs[:, task_classes[task_idx]]
                
                # 라벨을 0부터 시작하도록 변경
                adjusted_labels = labels - min(task_classes[task_idx])
                
                # Classification Loss 계산
                cls_loss = classification_criterion(student_logits_for_task, adjusted_labels)
                
                # Distillation Loss 계산
                distill_loss = torch.tensor(0.0, device=device)
                if config['kd_method'] == 'logit':
                    distill_loss += distillation_loss(student_logits_for_task, teacher_outputs, temperature)
                elif config['kd_method'] == 'attention':
                    # Teacher와 Student의 Feature Map 추출 (예: 첫 번째 Conv 레이어)
                    student_features = student_model.features[0](images)
                    teacher_features = teacher_model.features[0](images)
                    distill_loss += attention_transfer_loss(student_features, teacher_features)
                elif config['kd_method'] == 'feature':
                    # 중간 레이어의 Feature Map을 비교 (예: 두 번째 Conv 레이어)
                    student_features = student_model.features[2](images)
                    teacher_features = teacher_model.features[2](images)
                    distill_loss += nn.MSELoss()(student_features, teacher_features)
                elif config['kd_method'] == 'cat':
                    # Class Attention Transfer (간단화된 예시)
                    student_features = student_model.features[0](images)
                    teacher_features = teacher_model.features[0](images)
                    distill_loss += attention_transfer_loss(student_features, teacher_features)
                
                # 데이터 리플레이 추가 (Exemplar Replay)
                if config['replay_method'] == 'exemplar' and exemplar_store is not None:
                    replay_dataset = exemplar_store.get_replay_dataset()
                    if replay_dataset:
                        replay_loader = DataLoader(replay_dataset, batch_size=batch_size, shuffle=True)
                        for replay_images, replay_labels in replay_loader:
                            replay_images = replay_images.to(device)
                            replay_labels = replay_labels.to(device)
                            replay_outputs = student_model(replay_images)
                            # Replay Loss 계산
                            replay_cls_loss = classification_criterion(replay_outputs, replay_labels)
                            distill_loss += replay_cls_loss  # 단순 예시
                
                # Old 데이터 접근 제한 시 Data-Free KD 적용
                if config['data_free_kd'] is not None:
                    if config['data_free_kd'] in ['data_free', 'pseudo']:
                        # Pseudo Data Generation
                        pseudo_data = generate_pseudo_data(generator, num_samples=batch_size, noise_dim=noise_dim)
                        #print(f"Pseudo data shape: {pseudo_data.shape}")  # 디버깅 출력 추가
                        pseudo_outputs = teacher_model(pseudo_data)
                        #print(f"Pseudo outputs shape: {pseudo_outputs.shape}")  # 디버깅 출력 추가
                        student_pseudo_outputs = student_model(pseudo_data)
                        #print(f"Student pseudo outputs shape: {student_pseudo_outputs.shape}")  # 디버깅 출력 추가
                        
                        # Student 모델의 현재 태스크 클래스에 해당하는 출력만 선택
                        student_pseudo_outputs_task = student_pseudo_outputs[:, task_classes[task_idx]]
                        #print(f"Student pseudo outputs task shape: {student_pseudo_outputs_task.shape}")  # 디버깅 출력 추가
                        
                        # Distillation Loss 계산
                        distill_loss += distillation_loss(student_pseudo_outputs_task, pseudo_outputs, temperature)
                    elif config['data_free_kd'] == 'functional':
                        # Functional Regularization
                        functional_loss = functional_regularization_loss(student_logits_for_task, teacher_outputs)
                        distill_loss += functional_loss
                
                # 이전 태스크의 Distillation Loss 합산 (Catastrophic Forgetting 방지)
                if config['kd_method'] in ['attention', 'cat'] and config['kd_method'] and len(previous_teacher_models) > 1:
                    for prev_task_idx, prev_teacher_model in enumerate(previous_teacher_models[:-1]):
                        prev_teacher_model.eval()
                        with torch.no_grad():
                            prev_teacher_outputs = prev_teacher_model(images)
                        # Attention Transfer
                        student_features_prev = student_model.features[0](images)
                        prev_teacher_features = prev_teacher_model.features[0](images)
                        distill_loss += attention_transfer_loss(student_features_prev, prev_teacher_features)
                
                # Regularization Loss 계산
                if config['reg_method'] == 'EWC' and ewc is not None:
                    reg_loss = ewc.penalty(student_model)
                else:
                    reg_loss = torch.tensor(0.0, device=device)
                
                # 전체 Loss 계산
                loss = alpha * distill_loss + (1 - alpha) * cls_loss + reg_loss
                loss.backward()
                optimizer.step()
                train_total_loss += loss.item()
                
                # 정확도 계산
                _, predicted = torch.max(student_logits_for_task.data, 1)
                train_total_samples += labels.size(0)
                train_total_correct += (predicted == adjusted_labels).sum().item()
            
            train_avg_loss = train_total_loss / len(train_loader)
            train_accuracy = 100 * train_total_correct / train_total_samples
            
            # 테스트 정확도 계산
            student_model.eval()
            test_total_correct = 0
            test_total_samples = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = student_model(images)
                    student_logits_for_task = outputs[:, task_classes[task_idx]]
                    adjusted_labels = labels - min(task_classes[task_idx])
                    _, predicted = torch.max(student_logits_for_task.data, 1)
                    test_total_samples += labels.size(0)
                    test_total_correct += (predicted == adjusted_labels).sum().item()
            test_accuracy = 100 * test_total_correct / test_total_samples
            
            # 초기 정확도 저장
            if len(initial_accuracies) <= task_idx:  # 이미 저장되지 않은 경우만 추가
                initial_accuracies.append(test_accuracy)

            # **삽입 위치: Forgetting 측정**
            if task_idx > 0:  # 첫 번째 태스크는 Forgetting 측정 대상이 아님
                for prev_task_idx, prev_test_dataset in enumerate(task_test_datasets[:task_idx]):
                    prev_test_loader = DataLoader(prev_test_dataset, batch_size=batch_size, shuffle=False)
                    prev_total_correct = 0
                    prev_total_samples = 0
                    with torch.no_grad():
                        for images, labels in prev_test_loader:
                            images = images.to(device)
                            labels = labels.to(device)
                            outputs = student_model(images)
                            student_logits_for_task = outputs[:, task_classes[prev_task_idx]]
                            adjusted_labels = labels - min(task_classes[prev_task_idx])
                            _, predicted = torch.max(student_logits_for_task.data, 1)
                            prev_total_samples += labels.size(0)
                            prev_total_correct += (predicted == adjusted_labels).sum().item()
                    prev_test_accuracy = 100 * prev_total_correct / prev_total_samples
                    forgetting = max(0, initial_accuracies[prev_task_idx] - prev_test_accuracy)
                    forgetting_metrics.append(forgetting)
  
                # **루프 종료 후 한 번만 Forgetting 평균 출력**
                if forgetting_metrics:
                    avg_forgetting = np.mean(forgetting_metrics)
                    print(f"Average Forgetting so far: {avg_forgetting:.4f}")
                else:
                    print("Average Forgetting so far: None")

            print(f"Student Epoch [{epoch + 1}/{num_epochs}], Loss: {train_avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")
        
        # 태스크 완료 후 EWC 업데이트 (다음 태스크를 위해 현재 태스크의 지식을 저장)
        if config['reg_method'] == 'EWC':
            if ewc is None:
                ewc = EWC(copy.deepcopy(student_model), task_train_dataset, lambda_=lambda_ewc)
            else:
                ewc._add_model(copy.deepcopy(student_model), task_train_dataset)

    results_store.append({
        "description": config["description"],
        "train_loss": train_avg_loss,  # 마지막 태스크의 학습 손실
        "train_acc": train_accuracy,  # 마지막 태스크의 학습 정확도
        "test_acc": test_accuracy,    # 마지막 태스크의 테스트 정확도
        "forgettings": np.mean(forgetting_metrics) if forgetting_metrics else 0  # 평균 Forgetting
    })

    print(f"[Experiment {experiment_id + 1}] Results stored successfully.")

# 실험 설정 자동 생성
num_teachers_options = [2, 5]
softmax_options = [True, False]
kd_method_options = ['logit', 'attention', 'feature', 'cat', None]
reg_method_options = ['EWC', 'SI', None]
replay_method_options = ['generative', 'exemplar', None]
data_free_kd_options = ['data_free', 'pseudo', 'functional', None]
temperature_options = [1.0, 2.0, 5.0]

# 실험 설정 생성 및 실행
for num_teachers in num_teachers_options:
    for temperature in temperature_options:
        for softmax in softmax_options:
            for kd_method in kd_method_options:
                for reg_method in reg_method_options:
                    for replay_method in replay_method_options:
                        for data_free_kd in data_free_kd_options:
                            # KD가 사용되지 않으면 다른 관련 설정도 사용하지 않음
                            if kd_method is None and (replay_method is not None or reg_method is not None or data_free_kd is not None):
                                continue
                            if kd_method == 'logit' and not softmax:
                                continue
                            description = (
                                f"Teachers: {num_teachers}, Temperature: {temperature}, "
                                f"Softmax: {'On' if softmax else 'Off'}, KD: {kd_method}, "
                                f"Reg: {reg_method}, Replay: {replay_method}, Data-Free KD: {data_free_kd}"
                            )
                            config = {
                                'description': description,
                                'num_teachers': num_teachers,
                                'temperature': temperature,
                                'softmax': softmax,
                                'kd_method': kd_method,  # 'logit', 'attention', 'feature', 'cat'
                                'reg_method': reg_method,  # 'EWC', 'SI', None
                                'replay_method': replay_method,  # 'generative', 'exemplar', None
                                'data_free_kd': data_free_kd  # 'data_free', 'pseudo', 'functional', None
                            }
                            try:
                                run_experiment(config, len(results_store), results_store)
                            except Exception as e:
                                print(f"Experiment failed for config: {description}\nError: {e}")



# 실험 디렉토리 생성
results_dir = "experiment_results"
os.makedirs(results_dir, exist_ok=True)

# 최종 결과를 DataFrame으로 저장
df_results_store = pd.DataFrame(results_store)
results_dir = "experiment_results"
os.makedirs(results_dir, exist_ok=True)  # 디렉토리 생성은 한 번만 수행
results_store_file_path = os.path.join(results_dir, "results_store.csv")
df_results_store.to_csv(results_store_file_path, index=False)

print(f"\nAll experiments completed. Results saved to '{results_store_file_path}'")

# 시각화 수행
def visualize_results(df):
    if 'test_accuracy' not in df.columns:
        print("No 'test_accuracy' column found. Skipping visualization.")
        return
    plt.figure(figsize=(12, 6))
    plt.bar(df.index, df['test_accuracy'], color='skyblue')
    plt.xlabel('Experiment ID')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Experiment Results')
    plt.xticks(df.index, df['description'], rotation=45, ha='right', fontsize=8)
    plt.ylim(0, 100)
    for idx, acc in enumerate(df['test_accuracy']):
        plt.text(idx, acc + 1, f"{acc:.2f}%", ha='center', fontsize=8)
    plt.tight_layout()
    plt.show()

# 시각화 호출
visualize_results(df_results_store)

# 디버그 모드 설정
debug_mode = False

if debug_mode and ewc is not None and ewc.params:
    for n, p in student_model.named_parameters():
        if p.requires_grad:
            if n in ewc.params[0]:
                if not torch.allclose(p, ewc.params[0][n]):
                    print(f"[EWC] Parameter '{n}' mismatch after deepcopy.")
            else:
                print(f"[EWC] Parameter '{n}' not found in EWC params.")