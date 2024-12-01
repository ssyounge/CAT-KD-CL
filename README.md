아래는 깃허브 README 형식으로 변환한 내용입니다. 각 섹션을 깔끔하게 정리하여 바로 사용할 수 있도록 작성하였습니다.

---

# **Class Attention Transfer 기반 Continual Learning**

## **1. 연구 목적**
- Continual Learning 환경에서 이전 태스크와 현재 태스크를 동시에 해결할 수 있는 Knowledge Distillation(KD) 기법 개발.
- Class Attention Transfer(CAT)를 활용하여 Forgetting 감소 및 Accuracy 향상을 목표로 함.
- MNIST 데이터셋을 기반으로 CAT의 효과와 주요 하이퍼파라미터(`lambda_ewc`, `temperature`, `exemplar_store_size`, `generator_num_epochs`)의 영향을 분석.

---

## **2. 실험 설정**
### **데이터셋**
- **MNIST**: 0-5 클래스는 이전 태스크, 6-9 클래스는 현재 태스크로 설정.

### **모델**
- **Teacher 모델**: 이전 태스크에서 학습한 고성능 모델.
- **Student 모델**: 현재 태스크를 학습하며 Teacher의 지식을 Distillation.

### **평가지표**
- **Forgetting (`total_forgetting`)**: 이전 태스크 성능의 감소율.
- **Accuracy (`test_acc`)**: 이전 및 현재 태스크에서의 분류 정확도.

### **주요 변수**
- `lambda_ewc`: Regularization 강도.
- `temperature`: Softmax Temperature 조정.
- `exemplar_store_size`: Exemplar Replay 크기.
- `generator_num_epochs`: Generator 학습 Epoch.

---

## **3. 주요 결과**

### **1. Lambda EWC**
- **효과**: Forgetting 감소에 가장 효과적인 변수.
- **결과**:
  - Lambda 값이 증가할수록 Forgetting 감소 효과가 뚜렷.
  - 1000~5000 범위에서 Accuracy와 Forgetting의 균형이 가장 이상적.
  - 너무 높은 값(10000)에서는 Accuracy가 감소.

---

### **2. Temperature**
- **효과**: 낮은 Temperature 값에서 안정적인 성능.
- **결과**:
  - 낮은 값(1)에서 Accuracy가 가장 높고 Forgetting이 최소화.
  - 값이 증가하면 Forgetting이 증가하고 Accuracy가 감소.

---

### **3. Exemplar Store Size**
- **효과**: Store Size 증가로 Forgetting 감소와 Accuracy 향상.
- **결과**:
  - Store Size가 커질수록 Forgetting이 감소하고 Accuracy가 상승.
  - Store Size가 500 이상일 때 성능이 안정화.

---

### **4. Generator Epochs**
- **효과**: Accuracy 향상에 기여.
- **결과**:
  - Epochs가 증가할수록 Accuracy는 상승.
  - Forgetting은 Epochs가 많아질수록 소폭 증가.

---

### **5. Replay Method와 데이터 증강**
- **효과**: MNIST 데이터셋에서의 제한적 효과.
- **결과**:
  - Replay Method(Exemplar, Generative)는 MNIST 데이터셋에서 Forgetting 감소에 비효율적.
  - 복잡한 데이터셋에서는 추가적인 효과 검증 필요.

---

### **6. Teacher와 Student 모델 깊이**
- **결과**:
  - MNIST와 같은 단순 데이터셋에서 모델 깊이를 조절하면 과적합 발생.
  - 단순한 모델 구성으로도 충분한 성능 발휘.


---

## **4. 결론 및 제안**

### **결론**
1. **Lambda EWC**와 **Temperature**는 Forgetting 감소와 Accuracy 향상에 중요한 변수로 작용.
2. 적절한 설정(`lambda_ewc=1000~5000`, `temperature=1`)으로 안정적인 성능 확보.
3. Replay Method는 MNIST에서는 효과가 제한적이며, 복잡한 데이터셋에서 추가 검증이 필요.

---


### **제안**
1. **복잡한 데이터셋 실험**:
   - CIFAR-10, CIFAR-100 등 고복잡도 데이터셋에서 Replay Method 및 CAT 성능 검증.
2. **상호작용 분석**:
   - Lambda EWC, Temperature, Exemplar Store Size 간의 상호작용 효과 추가 분석.
3. **학습 효율성 검토**:
   - Teacher와 Student 모델의 크기와 구조가 복잡한 데이터셋에서도 동일한 효과를 보이는지 확인.

---

## **첨부 시각화**
- 각 변수(`lambda_ewc`, `temperature`, `exemplar_store_size`, `generator_num_epochs`)가 Test Accuracy와 Total Forgetting에 미치는 영향을 그래프로 정리.
- 주요 시각화:
  1. Lambda EWC
![d4a6a46a-e258-474a-a33e-28e8a952fda5](https://github.com/user-attachments/assets/7f6cafa2-fdcd-4991-be87-7c0b4438000d)
![bc0bcce0-3d3f-4dd7-a188-3eb9b79e9c9c](https://github.com/user-attachments/assets/79ebbd78-bfbb-4759-9ac9-36f2e17137c9)
  3. Temperature
  4. Exemplar Store Size
  5. Generator Epochs

