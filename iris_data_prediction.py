from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

iris=load_iris()
iris_data=iris.data
iris_label=iris.target

iris_df=pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label']=iris.target

# 데이터 분리
x_train,x_test,y_train,y_test=train_test_split(iris_data,iris_label,test_size=0.2,random_state=11)

# 모델 학습
dt_clf=DecisionTreeClassifier(random_state=11)
dt_clf.fit(x_train,y_train)

# 예측 수행
prediction=dt_clf.predict(x_test)

# 평가
print(accuracy_score(y_test,prediction))