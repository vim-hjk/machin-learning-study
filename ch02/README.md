Chapter02 : Scikit-Learn으로 시작하는 머신러닝
=====
## 붓꽃 품종 예측하기
 - `sklearn.datasets` : scikit-learn에서 자체적으로 제공하는 데이터 세트를 생성하는 모듈
 - `sklearn.tree` : 트리 기반 ML 알고리즘을 구현한 클래스의 모임
 - `sklearn.model_selection` : 학습데이터와 검증 데이터, 예측 데이터로 데이터를 분리하거나 최적의 hyperparameter로 평가하기 위한 다양한 모듈의 모임
 - iris dataset, DecisionTreeClassifier 사용
 - training set과 test set을 나눌 때 train_test_split() API를 사용
 - [1_iris_classification.py](https://github.com/vim-hjk/machin-learning-study/blob/main/ch02/1_iris_classification.py)

 -----
 ## scikit-learn의 기반 프레임워크 익히기
 ### 1. Estimator 이해 및 fit(). predict() 메서드
 - 분류 알고리즘인 Classifier와 회귀 알고리즘은 Regressor를 지원한다.
 - `fit()`과 `predict()`를 통한 간단한 학습 및 예측이 가능하다.
 - 이러한 Classifier와 Regressor를 통칭해서 **Estimator**라고 한다.
- PCA, Clustering, Feature Extraction 등을 구현한 class 역시 `fit()`과 `transform()`을 적용한다. 하지만 여기서는 학습과 예측이 아니라 데이터의 사전 구조를 맞추는 작업과 실제 작업을 의미한다.

### 2. 주요 모듈
| 분류                              | 모듈명                     | 설명                                                                                                                                                    |
|-----------------------------------|----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| 예제 데이터                       | `sklearn.datasets`           | 사이킷런에 내장되어 예제로 제공하는 데이터 세트                                                                                                         |
| 피처처리                          | `sklearn.preprocessing`      | 데이터 전처리에 필요한 다양한 가공 기능 제공(문자열을 숫자형 코드 값으로 인코딩, 정규화, 스케일링 등)                                                   |
|                                   | `sklearn.feature_selection`  | 알고리즘에 큰 영향을 미치는 피처를 우선순위대로 셀렉션 작업을 수행하는 다양한 기능 제공                                                                 |
|                                   | `sklearn.feature_extraction` | 텍스트 데이터나 이미지 데이터의 벡터화된 피처를 추출하는데 사용됨.                                                                                      |
|                                   |                            | 예를 들어 텍스트 데이터에서 Count Vectorizer나                                                                                                          |
|                                   |                            | Tf-Idf Vectorizer 등을 생성하는 기능 제공.                                                                                                              |
|                                   |                            | 텍스트 데이터의 피처 추출은 `sklearn.feature_extraction.text` 모듈에, 이미지 데이터의 피처 추출은 `sklearn.feature_extraction.image` 모듈에 지원 API가 있음 |
| 피처 처리 & 차원 축소             | `sklearn.decomposition`      | 차원 축소와 관련한 알고리즘을 지원하는 모듈이다. PCA, NMF, Truncated SVD 등을 통해 차원 축소 기능을 수행할 수 있다.                                     |
| 데이터 분리, 검증 & 파라미터 튜닝 | `sklearn.model_selection`    | 교차 검증을 위한 학습용/테스트용 분리, 그리드 서치(Grid Search)로 최적 파라미터 추출 등의 API 제공                                                      |
| 평가                              | `sklearn.metrics`            | 분류, 회귀, 클러스터링, 페어와이즈(Pairwise)에 대한 다양한 성능 측정 방법 제공                                                                          |
|                                   |                            | Accuracy, Precision, Recall, ROC-AUC, RMSE 등 제공                                                                                                      |
| ML 알고리즘                       | `sklearn.ensemble`           | 앙상블 알고리즘 제공                                                                                                                                    |
|                                   |                            | 랜덤 포레스트, 에이다 부스트, 그래디언트 부스팅 등을 제공                                                                                               |
|                                   | `sklearn.linear_model`       | 주로 선형 회귀, 릿지(Ridge), 라쏘(Lasso) 및 로지스틱 회귀 등 회귀 관련 알고리즘을 지원. 또한 SGD(Stochastic Gradient Desccent) 관련 알고리즘도 제공     |
|                                   | `sklearn.naïve_bayes`        | 나이브 베이즈 알고리즘 제공. 가우시안 NB. 다항 분포 NB 등                                                                                               |
|                                   | `sklearn.neighbors`          | 최근접 이웃 알고리즘 제공. K-NN(K-Nearest Neighborhood) 등                                                                                              |
|                                   | `sklearn.svm`                | 서포트 벡터 머신 알고리즘 제공                                                                                                                          |
|                                   | `sklearn.tree`               | 의사 결정 트리 알고리즘 제공                                                                                                                            |
|                                   | `sklearn.cluster`            | 비지도 클러스터링 알고리즘 제공                                                                                                                         |
|                                   |                            | (K-평균, 계층형, DBSCAN 등)                                                                                                                             |
| 유틸리티                          | `sklearn.pipeline`           | 피처 처리 등의 변환과 ML 알고리즘 학습, 예측 등을 함께 묶어서 실행할 수 있는 유틸리티 제공                                                              |
 
### 3. Model Selection Module
- 학습/테스트 데이터 셋 분리
    - [3_model_selection.py](https://github.com/vim-hjk/machin-learning-study/blob/main/ch02/3_model_selection.py)

- Cross-Validation
    - 학습 데이터가 적을 때 주로 사용
    - 고정된 train data와 test 데이터로 평가를 하게 되다보면 테스트 데이터에만 최적의 성능을 발휘할 수 있도록 편향되게 모델을 유도하는 경향이 생기게 됨 → Overfitting
    - Overfitting을 방지하기 위한 학습 기법
    - 본고사를 치르기 전에 모의고사를 여러번 보는 방법

- K-Fold Cross validation
    - [Link](https://medium.com/the-owl/k-fold-cross-validation-in-keras-3ec4a3a00538)
    - ![img](https://miro.medium.com/max/601/1*PdwlCactbJf8F8C7sP-3gw.png)
    
    - 학습 데이터 세트와 검증 데이터 세트를 점진적으로 변경하면서 학습과 검증을 수행하는 방법

- Stratified K-Fold
    - 불균형한 분포도를 가진 label 데이터 집합을 위한 방식으로 K-Fold Cross validation 방법을 쓸 때 가장 보편적으로 사용된다.
    - 원본 label 분포를 먼저 고려한 뒤 이 분포와 동일하게 학습과 검증 데이터 세트를 분배한다.
    - [5_stratified_k_fold.py](https://github.com/vim-hjk/machin-learning-study/blob/main/ch02/5_stratified_k_fold.py)
    - `cross_val_score()` API를 사용하면 편리하게 사용할 수 있다.

- GridSearchCV
    - cross validation과 최적 hyperparameter tuning을 한 번에 수행하는 방법
    - dictionary type으로 실험을 위한 파라미터 셋팅을 저장한 후 `param_grid=`로 전달하면 cross validation을 통해 최적의 hyperparameter를 찾아준다.
    - `.best_params` : 최적 파라미터 값
    - `.best_score_` : 최고 정확도

### 4. 데이터 전처리
- Data Encoding
    - Label Encoding
        - category feature를 코드형 숫자 값으로 변환하는 작업
        - `LabelEncoder()` class로 구현
        - `fit()`과 `transform()`을 호출해 수행한다.
    - One-Hot Encoding
        - feature 값 유형에 따라 새로운 feature를 추가해 고유 값에 해당하는 column에만 1을 표시하고 나머지는 0으로 표시하는 방법
        - `OneHotEncoder()`사용
        - **변환하기 전에 모든 문자열 값이 숫자형 값으로 먼저 변환돼야 하며, 입력값으로는 2차원 데이터가 필요하다.**

- Feature Scaling and Normalization
    - 서로 다른 변수의 값 범위를 일정한 수준으로 맞추는 작업
    - 대표적으로 Standardization과 Normalization이 있다.
    - Standardization(표준화)
        - <img src="https://latex.codecogs.com/gif.latex?x_{i}\_new=\frac{x_i-mean(x)}{stdev(x)}" />
        - 서로 다른 feature size를 통일 하기 위해 크기를 변환해주는 개념
        - 평균이 0이고 분산이 1인 가우시안 정규 분포를 가진 값으로 변환
        - 개별 데이터의 크기를 모두 똑같은 단위로 변경하는 것    

    - Normalization(정규화)
        - <img src="https://latex.codecogs.com/gif.latex?x_{i}\_new=\frac{x_i}{\sqrt{x_i^2+y_i^2+z_i^2}}" />
        - 선형대수에서의 정규화 개념이 적용
        - 개별 벡터의 크기를 맞추기 위해 변환하는 것을 의미
        - 개별 벡터를 모든 피처 벡터의 크기로 나눠 준다.

    - StandardScaler
        - 표준화 지원 class
        - `fit()`과 `transform()`사용

    - MinMaxScaler
        - 데이터 값을 0과 1사이의 범위 값으로 변환한다.(음수값이 있다면 -1에서 1값으로 변환)
        - 데이터의 분포가 가우시안 분포가 아닐 경우에 적용해 볼 수 있다.
        
    - **주의 사항**
        - `fit()`은 데이터 변환을 위한 기준 정보 설정을 적용하며, `transform()`은 이렇게 설정된 정보를 이용해 데이터를 변환한다. 즉, test data에는 `fit()`을 적용하지 않고 `transform()`만 수행해야한다. test data에 기준을 적용하면 train data와 기준 정보가 달라서 올바른 예측결과를 도출할 수 없다.

        - 가능한 전체 데이터의 scaling 변환을 적용한 뒤에 학습과 테스트 데이터로 분리하자
        - 그럴 수 없다면 test data에는 `fit()`을 사용하지 않는다.

- [8_feature_scaling_and_normalization.py](https://github.com/vim-hjk/machin-learning-study/blob/main/ch02/8_feature_scaling_and_normalization.py)

### 5. 타이타닉 생존자 예측
- [9_titanic.py](https://github.com/vim-hjk/machin-learning-study/blob/main/ch02/9_titanic.py) 참조


    
    