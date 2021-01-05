from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np

items=['TV','냉장고','전자레인지','컴퓨터','선풍기','선풍기','믹서','믹서']

# 먼저 숫자 값으로 변환을 위해 LabelEncoder로 변환합니다.

encoder=LabelEncoder()
encoder.fit(items)
labels=encoder.transform(items)

# 2차원 데이터로 변환합니다.
labels = labels.reshape(-1,1)

# one-hot encoding을 적용합니다.

oh_encoder=OneHotEncoder()
oh_encoder.fit(labels)
oh_encoder = oh_encoder.transform(labels)
print(oh_encoder.toarray())
print(oh_encoder.shape)