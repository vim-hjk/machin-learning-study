from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# 붓꽃 데이터 세트를 로딩한다.
iris = load_iris()

iris_data = iris.data

iris_label = iris.target
print('iris target 값 :', iris_label)
print('iris target 명 : ', iris.target_names)

# 붓꽃 데이터 세트 살펴보기
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label'] = iris.target
print(iris_df.head(3))

# trina, test 데이터로 나누기
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=11)

# 의사결정 트리 학습
dt_clf = DecisionTreeClassifier(random_state=11)
dt_clf.fit(X_train, y_train)

pred = dt_clf.predict(X_test)

# 학습 결과 출력
from sklearn.metrics import accuracy_score

print(f'예측 정확도 : {accuracy_score(y_test, pred):.4f}')

