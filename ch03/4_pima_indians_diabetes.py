from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Binarizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils


diabetes_data = pd.read_csv('diabetes.csv')
print(diabetes_data['Outcome'].value_counts())
print(diabetes_data.head(3))
print(diabetes_data.info())

# 피처 데이터 세트 X, 레이블 데이터 세트 y를 추출
# 맨 끝이 Outcome column으로 label 값임. column 위치 -1을 이용해 추출
X = diabetes_data.iloc[:, :-1]
y = diabetes_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=156, stratify=y)

# 로지스틱 회귀로 학습, 예측 및 평가 수행
lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
pred = lr_clf.predict(X_test)
pred_proba = lr_clf.predict_proba(X_test)[:, 1]

utils.get_clf_eval(y_test, pred, pred_proba)

pred_proba_c1 = lr_clf.predict_proba(X_test)[:, 1]
utils.precision_recall_curve_plot(y_test, pred_proba_c1)

print(diabetes_data.describe())
print(plt.hist(diabetes_data['Glucose'], bins=10))
plt.show()

# 0값을 검사할 feature 명 리스트
zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# 전체 데이터 건수
total_count = diabetes_data['Glucose'].count()

# feature별로 반복하면서 데이터 값이 0인 데이터 건수를 추출하고, 퍼센트 계산
for feature in zero_features:
    zero_count = diabetes_data[diabetes_data[feature] == 0][feature].count()
    print(f'{feature} 0 건수는 {zero_count}, 퍼센트는 {100*zero_count/total_count:.2f} %')


# zero_features 리스트 내부에 저장된 개별 feature들에 대해서 0값을 평균 값으로 대체
mean_zero_features = diabetes_data[zero_features].mean()
diabetes_data[zero_features] = diabetes_data[zero_features].replace(0, mean_zero_features)

# StandardScaler 클래스를 이용해 feature dataset에 일괄적으로 scailing 적용
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=156, stratify=y)

# 로지스틱 회귀로 학습, 예측 및 평가 수행
lr_clf.fit(X_train, y_train)
pred = lr_clf.predict(X_test)
pred_proba = lr_clf.predict_proba(X_test)[:, 1]

utils.get_clf_eval(y_test, pred, pred_proba)

thresholds = [0.3, 0.33, 0.36, 0.39, 0.42, 0.45, 0.28, 0.50]
pred_proba = lr_clf.predict_proba(X_test)
utils.get_eval_by_threshold(y_test, pred_proba[:, 1], thresholds)

# 임곗값을 0.48로 설정한 Binarizer 생성
binarizer = Binarizer(threshold=0.48)

# 위에서 구현한 lr_clf의 predict_proba() 예측 확률 array에서 1에 해당하는 column 값을 Binarizer로 반환
pred_th_048 = binarizer.fit_transform(pred_proba[:, 1].reshape(-1, 1))

utils.get_clf_eval(y_test, pred_th_048, pred_proba[:, 1])