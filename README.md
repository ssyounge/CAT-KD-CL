# **Class Attention Transfer 기반 Continual Learning**

## **1. 연구 목적**
- Continual Learning 환경에서 이전 태스크와 현재 태스크를 동시에 해결할 수 있는 Knowledge Distillation(KD) 기법 개발.
- Class Attention Transfer(CAT)를 활용하여 Forgetting 감소 및 Accuracy 향상을 목표로 함.
- MNIST 데이터셋을 기반으로 CAT의 효과와 주요 하이퍼파라미터(`lambda_ewc`, `temperature`, `exemplar_store_size`, `generator_num_epochs`)의 영향을 분석.

---

## **실험 설정**

### **1. 데이터셋**
- **MNIST**:
  - 0-5 클래스를 이전 태스크(Task 1), 6-9 클래스를 현재 태스크(Task 2)로 나누어 사용.
  - 각 태스크에 대해 별도의 데이터셋을 생성하며, `get_task_dataset` 함수를 사용해 클래스별로 데이터를 필터링.

---

### **2. 모델**
- **Teacher 모델**:
  - 이전 태스크를 학습한 고성능 모델.
  - 컨볼루션 레이어와 선형 분류기로 구성된 구조.
  - Feature Map을 통해 Class Attention Transfer(CAT)에 필요한 정보를 제공합니다.
- **Student 모델**:
  - 현재 태스크를 학습하는 모델.
  - Teacher 모델의 Feature Map과 출력을 Distillation을 통해 학습.

---

### **3. 평가지표**
- **Forgetting (`total_forgetting`)**:
  - 이전 태스크 성능의 감소율을 측정.
  - `forgetting_per_task`를 사용하여 각 태스크별 Forgetting 값을 계산.
- **Accuracy (`test_acc`)**:
  - 각 태스크에서의 최종 테스트 정확도를 측정.
  - 모델의 학습 성능을 평가하는 주요 지표.

---

### **4. 주요 변수**
- **`lambda_ewc`**:
  - Elastic Weight Consolidation(EWC)에서 Regularization 강도를 조절.
  - 값이 클수록 이전 태스크의 지식을 더 강하게 유지.
- **`temperature`**:
  - Knowledge Distillation에서 Softmax의 스무딩 효과를 조정하는 온도 매개변수.
  - 값이 클수록 출력 확률 분포가 부드러워짐.
- **`exemplar_store_size`**:
  - Exemplar Replay에서 저장할 샘플의 수.
  - Replay Method 사용 시 이전 태스크 데이터를 얼마나 많이 저장할지를 결정.
- **`generator_num_epochs`**:
  - Generative Replay에서 Generator의 학습 Epoch 수.
  - Generator가 충분히 학습되도록 제어.

---

### **5. 추가 변수**
- **Replay Method**:
  - `None`, `Exemplar`, `Generative` 중 선택.
  - 이전 태스크 데이터를 직접 Replay하거나 생성된 데이터를 활용.
- **Regularization Method (`reg_method`)**:
  - `None`, `EWC`, `SI` 중 선택.
  - EWC(Elastic Weight Consolidation) 또는 SI(Synaptic Intelligence)를 통해 파라미터의 안정성을 유지.

---

### **6. 생성 모델**
- **Generator**:
  - Pseudo Data를 생성하여 Generative Replay를 지원.
  - 노이즈를 입력으로 받아 MNIST 이미지와 유사한 데이터를 생성.

---

### **7. 하이퍼파라미터**
- **기본 설정**:
  - `batch_size=64`
  - `learning_rate=0.001`
  - `num_epochs=5`
  - `temperature=2.0`
  - `lambda_ewc=1000`
  - `exemplar_store_size=100`
  - `generator_num_epochs=10`
  - `noise_dim=100` (Generator 입력 노이즈 차원)


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
1. **상호작용 분석**:
   - Lambda EWC, Temperature, Exemplar Store Size 간의 상호작용 효과 추가 분석.
2. **학습 효율성 검토**:
   - Teacher와 Student 모델의 크기와 구조가 복잡한 데이터셋에서도 동일한 효과를 보이는지 확인.

---

## **첨부 시각화**
- 각 변수(`lambda_ewc`, `temperature`, `exemplar_store_size`, `generator_num_epochs`)가 Test Accuracy와 Total Forgetting에 미치는 영향을 그래프로 정리.
- 주요 시각화:
  1. Lambda EWC
![d4a6a46a-e258-474a-a33e-28e8a952fda5](https://github.com/user-attachments/assets/7f6cafa2-fdcd-4991-be87-7c0b4438000d)
![bc0bcce0-3d3f-4dd7-a188-3eb9b79e9c9c](https://github.com/user-attachments/assets/79ebbd78-bfbb-4759-9ac9-36f2e17137c9)

  2. Temperature
![4b564f5e-bf93-4680-93e8-92ca39a020f7](https://github.com/user-attachments/assets/7ba57657-d6f1-457f-a6e0-644c4e575f40) 
![7cb1429d-9942-48e1-b1bf-b15f8de2a3ed](https://github.com/user-attachments/assets/c62f8dc0-2ece-43ee-94bc-a6f5d63279d9)

  3. Exemplar Store Size    
![e4dbe8b3-3b9e-4318-b4da-8aff6faa5500](https://github.com/user-attachments/assets/66a44bc5-a068-4c5f-b76e-97afb10f5e1b)
![5a7e21f1-c1d8-4abb-8f77-790e7f0cfdf0](https://github.com/user-attachments/assets/687b5264-e2ff-401b-bc80-490b623e30da)
 
     
  4. Generator Epochs
![1480b4a6-929f-4b42-87df-4fa59a7737c0](https://github.com/user-attachments/assets/814b7269-651b-4fdb-a620-adcbace755b1)
![caacf099-3dad-46b9-860f-ddee2c84a03c](https://github.com/user-attachments/assets/cfbcdd87-d04b-4178-a5f4-fbd51d7f5cd4)

