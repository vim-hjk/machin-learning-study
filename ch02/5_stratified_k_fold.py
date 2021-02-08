from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score

iris = load_iris()
iris_df = pd.DataFrame(data=iris, columns=iris.feature_names)
iris_df['label'] = iris.target
iris_df['label'].value_counts()


# Normal K-Fold
kfold = KFold(n_splits=3)
n_iter = 0
for train_index, test_index in kfold.split(iris_df):
    n_iter += 1
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print(f'## 교차 검증 : {n_iter}')
    print(f'학습 레이블 데이터 분포 : \n', label_train.value_counts())
    print(f'검증 레이블 데이터 분포 : \n', label_test.value_counts())


# stratified K-Fold
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=3)
n_iter = 0

for train_index, test_index in skf.split(iris_df, iris_df['label']):
    n_iter += 1
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print(f'## 교차 검증 : {n_iter}')
    print(f'학습 레이블 데이터 분포 : \n', label_train.value_counts())
    print(f'검증 레이블 데이터 분포 : \n', label_test.value_counts()) 


# Train using stratified K-Fold
dt_clf = DecisionTreeClassifier(random_state=156)

skf = StratifiedKFold(n_splits=3)
n_iter = 0
cv_accuracy = []

features = iris.data
label = iris.target

# StratifiedFold의 split() 호출 시 반드시 레이블 데이터 세트도 추가 입력 필요
for train_index, test_index in skf.split(features, label):
    # split()으로 반환된 인덱스를 이용해 학습용, 검증용 테스트 데이터 추출
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    # 학습 및 예측
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)

    # 반복 시마다 정확도 측정
    n_iter += 1
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print(f'\n#{n_iter} 교차 검증 정확도 : {accuracy}, 학습 데이터 크기 : {train_size}, 검증 데이터 크기 : {test_size}')
    print(f'#{n_iter} 검증 데이터 인덱스 : {test_index}')
    cv_accuracy.append(accuracy)

# 교차 검증별 정확도 및 평균 정확도 계산
print('\n## 교차 검증별 정확도 : ', np.round(cv_accuracy, 4))
print('\n## 평균 검증 정확도 : ', np.mean(cv_accuracy))

# cross_val_score 활용
from sklearn.model_selection import cross_val_score, cross_validate

scores = cross_val_score(dt_clf, features, label, scoring='accuracy', cv=3)
print('\n\n교차 검증별 정확도 : ', np.round(scores, 4))
print('평균 검증 정확도 : ', np.round(np.mean(scores), 4))