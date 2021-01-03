from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate

iris_data=load_iris()
dt_clf=DecisionTreeClassifier(random_state=156)

data=iris_data.data
label=iris_data.target

score = cross_val_score(dt_clf,data,label,scoring='accuracy',cv=3)

print('교차 검증별 정확도:', np.round(score,4))
print('평균 검증별 정확도:', np.round(np.mean(score), 4))