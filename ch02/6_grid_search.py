from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from pprint import pprint

iris_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=121)

dtree = DecisionTreeClassifier()

## 파라미터를 딕셔너리 형태로 설정
grid_parameters = {'max_depth' : [1, 2, 3], 'min_samples_split' : [2, 3]}

import pandas as pd 

# param_grid의 하이퍼 파라미터를 3개의 train, test set fold로 나누어 테스트 수행 설정.
### refit=True가 default임. True이면 가장 좋은 파라미터 설정으로 재학습시킴.
grid_dtree = GridSearchCV(dtree, param_grid=grid_parameters, cv=3, refit=True)
grid_dtree.fit(X_train, y_train)

# GridSearchCV 결과를 추출해 DataFrame으로 전환
scores_df = pd.DataFrame(grid_dtree.cv_results_)
pprint(scores_df[['params', 'mean_test_score', 'rank_test_score', 'split0_test_score', 'split1_test_score', 'split2_test_score']])

print('GridSearchCV 최적 파라미터 : ', grid_dtree.best_params_)
print(f'GridSearchCV 최고 정확도 : {grid_dtree.best_score_:.4f}')

# GridSearchCV의 refit으로 이미 학습된 estimator 반환
estimator = grid_dtree.best_estimator_


# GridSearchCV의 best_estimator_는 이미 최적 학습이 됐으므로 별도 학습이 필요없음
from sklearn.metrics import accuracy_score

pred = estimator.predict(X_test)
print(f'테스트 데이터 세트 정확도 : {accuracy_score(y_test, pred):.4f}')
