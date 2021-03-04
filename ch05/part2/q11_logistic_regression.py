import numpy as np
from scipy.special import expit
'''

- GD를 이용하여 Logistic regression 학습

'''
w = []
c = []
class LogisticRegression:
    def __init__(self, learning_rate=0.01, threshold=0.01, max_iterations=100000, fit_intercept=True, verbose=True, mode=3):
        self._learning_rate = learning_rate  # 학습 계수
        self._max_iterations = max_iterations  # 반복 횟수
        self._threshold = threshold  # 학습 중단 계수
        self._fit_intercept = fit_intercept  # 절편 사용 여부를 결정
        self._verbose = verbose  # 중간 진행사항 출력 여부
        self._mode = mode

    # theta(W) 계수들 return
    def get_coeff(self):
        return self._W

    # 절편 추가
    def add_intercept(self, x_data):
        intercept = np.ones((x_data.shape[0], 1))
        return np.concatenate((intercept, x_data), axis=1)

    # 시그모이드 함수(로지스틱 함수)
    def sigmoid(self, z):
        return expit(z)

    def cost(self, h, y):
        epsilon = 1e-7
        if self._mode == 0:
            return (-y * np.log(h + epsilon) - (1 - y) * np.log((1 - h) + epsilon)).mean()
        if self._mode == 1 :
            return np.mean(np.square(h - y))

    def fit(self, x_data, y_data):
        num_examples, num_features = np.shape(x_data)

        if self._fit_intercept:
            x_data = self.add_intercept(x_data)

        # weights initialization
        self._W = np.zeros(x_data.shape[1])

        for i in range(self._max_iterations):
            z = np.dot(x_data, self._W)
            hypothesis = self.sigmoid(z)

            # 실제값과 예측값의 차이
            diff = hypothesis - y_data
            
            # cost 함수
            cost = self.cost(hypothesis, y_data)
            # cost = np.mean(np.square(y_data - hypothesis))
            # cost = (-y_data * np.log(hypothesis) - (1 - y_data) * np.log(1 - hypothesis)).mean()
            c.append(cost)
            # cost 함수의 편미분 : transposed X * diff / n
            # 증명 : https://stats.stackexchange.com/questions/278771/how-is-the-cost-function-from-logistic-regression-derivated
            gradient = np.dot(x_data.transpose(), diff) / num_examples
            # gradient에 따라 theta 업데이트
            self._W -= self._learning_rate * gradient
            w.append(np.sum(self._W))
            # 판정 임계값에 다다르면 학습 중단
            # if cost < self._threshold:
            #    return False

            # 100 iter 마다 cost 출력
            if (self._verbose == True and i % 10000 == 0):
                print('cost :', cost)

    def predict_prob(self, x_data):
        if self._fit_intercept:
            x_data = self.add_intercept(x_data)

        return self.sigmoid(np.dot(x_data, self._W))

    def predict(self, x_data):
        # 0,1 에 대한 판정 임계값은 0.5 -> round 함수로 반올림
        return self.predict_prob(x_data).round()


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

cancer = load_breast_cancer()

# StandardScaler( )로 평균이 0, 분산 1로 데이터 분포도 변환
scaler = StandardScaler()
data_scaled = scaler.fit_transform(cancer.data)

X_train , X_test, y_train , y_test = train_test_split(data_scaled, cancer.target, test_size=0.2, random_state=31)

fig, axs = plt.subplots(1,2, figsize=(20, 40), sharex=True, sharey=True)

# 로지스틱 회귀를 이용하여 학습 및 예측 수행. 
lr_clf = LogisticRegression(mode=1)
lr_clf.fit(X_train, y_train)
axs.flat[0].plot(w, c, color='r')
lr_preds = lr_clf.predict(X_test)

print(f'mse accuracy: {accuracy_score(y_test, lr_preds):0.3f}')
print(f'mse roc_auc: {roc_auc_score(y_test , lr_preds):0.3f}')

w = []
c = []
lr_clf2 = LogisticRegression(mode=0)
lr_clf2.fit(X_train, y_train)
axs.flat[1].plot(w, c, color='b')
lr_preds2 = lr_clf.predict(X_test)
# accuracy와 roc_auc 측정
print(f'log loss accuracy: {accuracy_score(y_test, lr_preds2):0.3f}')
print(f'log loss roc_auc: {roc_auc_score(y_test , lr_preds2):0.3f}')
for ax in axs:
    plt.setp(ax, xlabel='Weight')
    plt.setp(ax, ylabel='Cost')
plt.show()

