# **README: Continual Learning with Class Attention Transfer-Based Knowledge Distillation**

## **실험 목적**

본 실험의 목적은 **CVPR 논문**에서 제안된 Class Activation Mapping(CAM) 정보를 활용한 Knowledge Distillation(KD) 기법의 성능을 평가하고, 이를 Continual Learning 환경에 적용하여 **Catastrophic Forgetting** 문제를 완화하는 것입니다.

특히, **Class Attention Transfer(CAT) 기반 KD**를 통해 이전 태스크의 정보를 보존하면서 현재 태스크의 성능을 최적화하는 방식을 검증합니다. 이를 통해 이전과 현재 태스크를 동시에 해결할 수 있는 모델 학습 가능성을 입증하고자 합니다.

본 실험은 MNIST 데이터셋 기반의 Continual Learning 설정에서 다음을 중점적으로 분석합니다:
- CAT 기반 KD 기법이 기존 KD 방식(Logit Matching, Feature Matching 등) 대비 성능을 얼마나 효과적으로 향상시키는지.
- 이전 Teacher 모델과 현재 Teacher 모델의 CAM 정보를 동시에 활용하여 Student 모델이 **모든 태스크에 대해 높은 성능**을 유지할 수 있는지.

이를 통해 **CAT 기반 KD** 기법이 Continual Learning 환경에서 Catastrophic Forgetting 문제를 해결할 수 있는 **효과적인 접근법**임을 실증합니다.

---

## **실험 구조**

### **1. 데이터셋 및 태스크 구성**
- **데이터셋**: MNIST (10개 클래스의 손글씨 숫자 이미지).
- **태스크 구성**: 총 5개의 태스크, 각 태스크는 2개의 클래스 포함.
  - **예시**:
    - Task 1: 클래스 0, 1.
    - Task 2: 클래스 2, 3.
    - ...
    - Task 5: 클래스 8, 9.

---

### **2. 모델 설계**
#### **Teacher 모델**
- 태스크별로 독립적으로 학습.
- **CAM 기반 지식**을 활용하여 Student 모델에 정보 전달.
- Teacher 모델 수 설정:
  - **num_teachers = 2**:  
    Teacher 1은 Task 1, 3, 5, Teacher 2는 Task 2, 4를 담당.
  - **num_teachers = 5**:  
    각 Teacher 모델은 하나의 태스크만 담당.

#### **Student 모델**
- 여러 Teacher 모델의 지식을 결합하여 모든 태스크를 해결.
- **CAM 정보를 활용한 Knowledge Distillation**을 통해 학습.
- **목표**: 이전 태스크와 현재 태스크의 정보를 모두 보존하며 최적의 성능을 달성.

---

### **3. Knowledge Distillation(KD) 기법**
#### **Class Attention Transfer (CAT)**:
- Teacher와 Student 간 Attention Map 비교를 기반으로 Distillation Loss를 계산.
- Teacher의 CAM 정보와 Student의 Feature Map을 비교하여 성능 최적화.

#### **기타 KD 기법**:
- **Logit Matching**: Teacher와 Student의 로짓(Logits)을 비교.
- **Feature Matching**: Teacher와 Student의 중간 Feature Map 비교.

---

### **4. Regularization 및 Replay 기법**
- **Elastic Weight Consolidation (EWC)**:
  - 이전 태스크의 중요한 파라미터를 보존하기 위한 Regularization Loss.
- **Exemplar Replay**:
  - 이전 태스크 데이터를 일부 저장하고 학습 시 활용.
- **Data-Free KD**:
  - 가상 데이터를 생성하는 Generator를 활용해 Teacher-Student 학습에 사용.

---

## **실험 시나리오**

### **1. 학습 과정**
- 학습 순서: Task 1 → Task 2 → Task 3 → Task 4 → Task 5.
- **각 Task에서 진행**:
  - Teacher 모델은 해당 태스크 데이터를 학습.
  - Student 모델은 Teacher 모델의 CAM 정보를 활용하여 모든 태스크를 해결하도록 학습.

### **2. 평가 방식**
- **Test Accuracy**:
  - 각 Task 학습 완료 후, Student 모델의 성능을 해당 Task와 이전 Task에서 측정.
- **Catastrophic Forgetting**:
  - Forgetting Metric을 활용해 이전 Task 성능 감소율을 계산.

### **3. 실험 비교**
- CAT 기반 KD 기법 **사용** vs. **미사용**:
  - 기존 KD 방식(Logit Matching, Feature Matching 등)과 CAT 기반 KD의 성능 비교.
- **Regularization 및 Replay 기법 효과**:
  - EWC, Exemplar Replay, Data-Free KD 등과 결합 시 성능 변화 평가.

---

## **결과 측정**

### **1. 주요 지표**
- **Test Accuracy**:
  - Student 모델의 최종 성능을 태스크별로 측정.
- **Forgetting Metric**:
  - 이전 태스크에서 정확도 감소율 분석.
- **Loss 변화**:
  - Classification Loss, Distillation Loss, Regularization Loss 비교.

### **2. 결과 저장 및 시각화**
- **CSV 저장**: 모든 실험 결과를 `results_store.csv`에 저장.
- **시각화**: 각 실험의 테스트 정확도를 그래프로 비교.

---

## **실험 결과 해석**

1. **CAT-KD의 Continual Learning 효과 검증**  
   - CAT 기반 Knowledge Distillation이 Continual Learning에서 Catastrophic Forgetting 문제를 얼마나 효과적으로 줄이는지 확인.

2. **보조 기법과의 조합 효과 분석**  
   - EWC, Exemplar Replay, Data-Free KD 등의 기법을 CAT-KD와 결합했을 때 성능이 얼마나 향상되는지 평가.

3. **Teacher 모델 수와 태스크 분배의 영향 분석**  
   - Teacher 모델 수(num_teachers)가 Student 모델의 성능에 미치는 영향을 비교하여 최적의 Teacher-Student 구조 도출.

4. **MNIST 데이터셋 기반 CAT-KD 가능성 평가**  
   - CAT 기반 KD 기법이 MNIST 데이터셋에서 Continual Learning의 실질적 대안이 될 수 있음을 입증.
