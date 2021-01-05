from sklearn.preprocessing import LabelEncoder

items=['TV','냉장고','전자레인지','컴퓨터','선풍기','선풍기','믹서','믹서']

# LabelEncoder를 객체로 생성한 후, fit()과 transform()으로 레이블 인코딩 수행.

encoder=LabelEncoder()
encoder.fit(items)
labels=encoder.transform(items)
print(labels)
print(encoder.classes_)
print(encoder.inverse_transform([4,5,2,0,1,1,3,3]))