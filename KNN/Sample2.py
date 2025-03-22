import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#bir şirket işe eleman alacak eğitim seviyesi ve tecrübe yılına göre işe alım yapıyor.

data = [
    [0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0],
    [1, 0, 0], [1, 1, 0], [1, 2, 1], [1, 3, 1],
    [1, 4, 1], [1, 5, 1], [2, 0, 0], [2, 1, 1],
    [2, 2, 1], [2, 3, 1], [2, 4, 1], [2, 5, 1],
    [2, 6, 1], [2, 7, 1], [2, 8, 1], [2, 9, 1]
]

df = pd.DataFrame(data, columns=["school", "year", "hired"])

X= df[["school", "year"]]
y= df["hired"]

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=42)

k_values = range(1,16)
scores = []

#Accuracy score  
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k) #kaç komşuya bakılacak 
    model.fit(X_train,y_train)
    y_prediction =model.predict(X_test)
    accuracy =accuracy_score(y_test,y_prediction)
    scores.append(accuracy)
print(scores)

applicant = np.array([[1,2]])
prediction = model.predict(applicant)

print("Evet" if prediction[0]==1 else "Hayır")

#Cross Validation Arge konusu

from sklearn.model_selection import cross_val_score

k_values = range(1, 21)
scores = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(model, X_train, y_train, cv=5) # cv=5, 5 katlı çapraz doğrulama
    scores.append(score.mean())  # Ortalama doğruluk
    
best_k = k_values[scores.index(max(scores))]
print(f"En iyi k değeri: {best_k}")
