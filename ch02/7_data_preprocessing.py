from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 레이블 인코딩
items = ['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']

# LabelEncoder를 객체로 생성한 후, fit()과 transform()으로 레이블 인코딩 수행
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
print('인코딩 변환값 : ', labels)
print('인코딩 클래스 : ', encoder.classes_)
print('디코딩 원본값 : ', encoder.inverse_transform([4, 5, 2, 0, 1, 1, 3, 3]))

# One-hot encoding
import numpy as np

labels = labels.reshape(-1, 1) # 2차원 데이터로 변환

# One-hot encoding 적용
oh_encoder = OneHotEncoder() 
oh_encoder.fit(labels)
oh_labels = oh_encoder.transform(labels)
print('원-핫 인코딩 데이터')
print(oh_labels.toarray())
print('원-핫 인코딩 데이터 차원')
print(oh_labels.shape)

# pandas에서 지원하는 one-hot encoding
import pandas as pd
df = pd.DataFrame({'item' : ['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']})
print(pd.get_dummies(df))

