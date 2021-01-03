from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

iris=load_iris()
features=iris.data
label=iris.target
dt_clf=DecisionTreeClassifier(random_state=156)

# 5개의 폴드 세트로 분리하는 KFold 객체와 폴드 세트별 정확도를 담을 리스트 객체 생성.
kfold=KFold(n_splits=5)
cv_accuracy=[]

n_iter=0

# KFold 객체의 split()를 호출하면 폴드 별 학습용, 검증용 테스트의 로우 인덱스를 array로 반환
for train_index, test_index in kfold.split(features):
    # kfold.split()으로 반환된 인덱스를 이용해 학습용, 검증용 테스트 데이터 추출
    X_train,X_test = features[train_index], features[test_index]
    y_train,y_test = label[train_index], label[test_index]

    # 학습 및 예측
    dt_clf.fit(X_train,y_train)
    pred = dt_clf.predict(X_test)
    n_iter+=1

    # 반복 시마다 정확도 측정
    accuracy = np.round(accuracy_score(y_test,pred),4)
    print("%d번째 검증 정확도: %d" %(n_iter,accuracy))
    cv_accuracy.append(accuracy)

print("평균 검증 정확도: ", np.mean(cv_accuracy))