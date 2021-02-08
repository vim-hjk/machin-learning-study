from sklearn.datasets import load_iris

iris_data = load_iris()
print(type(iris_data))

keys = iris_data.keys()
print('붓꽃 데이터 세트의 키들 : ', keys)

print('\n feature_name의 type : ', type(iris_data.feature_names))
print('feature_name의 shape : ', len(iris_data.feature_names))
print(iris_data.feature_names)

print('\n target_names type : ', type(iris_data.target_names))
print('target_names shape : ', len(iris_data.target_names))
print(iris_data.target_names)

print('\n data type : ', type(iris_data.data))
print('feature_name의 shape : ', len(iris_data.data))
print(iris_data.data)

print('\n target type : ', type(iris_data.target))
print('target shape : ', len(iris_data.target))
print(iris_data.target)