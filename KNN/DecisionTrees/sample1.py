import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree
import random

def generateData(n=1000):
    data =[]
    for _ in range(n):
        age = random.randint(20,65)
        income = round(random.uniform(2.5,15.0),2)
        credit_score = random.randint(300,800)
        has_default = random.choice([0,1])
        approved = 1 if credit_score > 650 and income > 5 and not has_default else 0
        data.append([age, income, credit_score, has_default, approved])
    return pd.DataFrame(data, columns=["age", "income", "credit_score", "has_default", "approved"])

df = generateData()

data = pd.DataFrame({
    'age': [22, 35, 47, 52, 23, 40, 60, 30, 45, 29],
    'income': [3.5, 7.2, 5.0, 9.5, 2.8, 6.0, 4.2, 6.8, 5.5, 3.0],
    'credit_score': [650, 720, 580, 790, 610, 700, 640, 710, 690, 600],
    'has_default': [0, 0, 1, 0, 1, 0, 1, 0, 0, 1], #gecikme
    'approved': [1, 1, 0, 1, 0, 1, 0, 1, 1, 0]
})

X = data[["age", "income", "credit_score", "has_default"]]
y = data["approved"]

print(X)
print(y)

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train,y_train)

y_prediction = model.predict(X_test) 

print(accuracy_score(y_test,y_prediction))

plt.figure(figsize=(12,6))
tree.plot_tree(model, feature_names=X.columns, class_names=["Rejected", "Approved"], filled=True)
plt.title("Karar Ağacı Görselleştirmesi")
plt.show()

#bir kişinin durumunu test edin

new_person = pd.DataFrame([{
    "age": 34,
    "income": 6.5,
    "credit_score": 700,
    "has_default": 0
}])

prediction = model.predict(new_person)

# Tahmin sonucunu yazdır
result = "Approved" if prediction[0] == 1 else "Rejected"
print(f"Yeni kişinin başvuru durumu: {result}")
