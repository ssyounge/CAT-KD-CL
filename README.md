# **Continual Learning with Class Attention Transfer-Based Knowledge Distillation**

## **실험 목적**

본 실험의 목적은 **다수의 Teacher 모델**에서 학습한 지식을 **Knowledge Distillation(KD)**을 통해 Student 모델로 효과적으로 전달하여 **Continual Learning(CL)**을 가능하게 하는 방안을 모색하는 것입니다.

특히, 각 Teacher 모델이 학습한 클래스가 다르고, 이를 결합하여 **Catastrophic Forgetting** 문제를 완화하면서 새로운 태스크를 학습할 수 있는 방법을 검증합니다. 이 과정에서 **Class Attention Transfer(CAT) 기반 KD** 기법을 활용하여, Teacher 모델의 다양한 지식을 Student 모델에 효율적으로 통합하고자 합니다.

본 실험은 MNIST 데이터셋 기반 Continual Learning 환경에서 다음을 중점적으로 분석합니다:

- Teacher 모델의 분리된 지식을 Student 모델이 성공적으로 결합할 수 있는지.
- CAT 기반 KD가 Logit Matching 및 Feature Matching과 같은 기존 KD 기법 대비 얼마나 효과적인지.
- Teacher 모델의 수와 태스크 분배가 Student 모델 학습에 미치는 영향을 평가.

이를 통해 **Teacher 여러 개의 지식 결합을 통한 CL 가능성**과 **CAT 기반 KD 기법의 효과성**을 입증하고자 합니다.

---

## **실험 구조**

### **1. 데이터셋 및 태스크 구성**

- **데이터셋**: MNIST (10개 클래스의 손글씨 숫자 이미지).
- **태스크 구성**:
  - 총 **5개의 태스크**, 각 태스크는 **2개의 클래스**로 구성.
  - 예:
    - Task 1: 클래스 0, 1.
    - Task 2: 클래스 2, 3.
    - ...
    - Task 5: 클래스 8, 9.

---

### **2. 모델 설계**

#### **Teacher 모델**
- **각 Teacher 모델은 서로 다른 클래스에 대해 독립적으로 학습.**
- Teacher 모델 수 설정:
  - **num_teachers = 2**: 
    Teacher 1은 Task 1, 3, 5를 담당, Teacher 2는 Task 2, 4를 담당.
  - **num_teachers = 5**:
    각 Teacher는 하나의 태스크만 전담.

#### **Student 모델**
- 여러 Teacher 모델의 지식을 결합하여 **모든 태스크를 해결**하도록 학습.
- **목표**:
  - 이전 태스크의 지식을 잃지 않으면서도 새로운 태스크 학습 성능을 최적화.

---

### **3. Knowledge Distillation(KD) 기법**

#### **Class Attention Transfer (CAT)**:
- Teacher와 Student 간의 Attention Map 비교를 기반으로 Distillation Loss를 계산.
- Teacher의 CAM 정보와 Student의 Feature Map을 비교하여 지식 전달.

#### **기타 KD 기법**:
- **Logit Matching**: Teacher와 Student의 로짓(Logits)을 비교.
- **Feature Matching**: Teacher와 Student의 중간 Feature Map 비교.

---

### **4. Regularization 및 Replay 기법**

- **Elastic Weight Consolidation (EWC)**:
  - 이전 태스크에서 중요했던 파라미터를 보존하기 위한 Regularization 손실.
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
  - Student 모델은 Teacher 모델의 정보를 활용하여 모든 태스크를 해결하도록 학습.

### **2. 평가 방식**
- **Test Accuracy**:
  - 각 Task 학습 완료 후, Student 모델의 성능을 해당 Task와 이전 Task에서 측정.
- **Catastrophic Forgetting**:
  - Forgetting Metric을 활용해 이전 Task 성능 감소율을 계산.

### **3. 실험 비교**
- Teacher 모델 수(num_teachers = 2 vs. num_teachers = 5)에 따른 성능 차이 분석.
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

1. **Teacher-Student 기반 Continual Learning 가능성 검증**  
   - Teacher 여러 개의 분리된 지식을 Student 모델이 성공적으로 통합하여 CL을 가능하게 할 수 있는지 분석.

2. **CAT 기반 KD 기법의 효과 검증**  
   - CAT-KD가 Logit Matching, Feature Matching 등 기존 KD 기법 대비 얼마나 효과적인지 평가.

3. **Teacher 모델 수와 태스크 분배의 영향 분석**  
   - Teacher 모델 수(num_teachers)가 Student 모델 성능 및 학습 효율성에 미치는 영향을 비교하여 최적 구조 도출.

4. **Regularization 및 Replay 기법의 효과 검증**  
   - EWC, Exemplar Replay, Data-Free KD와 CAT-KD의 결합으로 성능이 향상되는지 확인.
