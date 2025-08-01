import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib

# 데이터 로딩
df = pd.read_csv("../data/pose_data.csv")
X = df.drop("label", axis=1)
y = df["label"]

# 학습
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

# 저장
joblib.dump(clf, "../model/gesture_model.pkl")

print(" 모델 저장 완료")
print(" 테스트 정확도:", clf.score(X_test, y_test))
