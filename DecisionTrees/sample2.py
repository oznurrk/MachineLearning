import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np
import random
import joblib
from flask import Flask,request,jsonify

#1 eğitim seviyesi 0=lise , 1 lisans, 2 YL
#2 tecrübe yılı
#3 hired?
def generateData(n=1000):
    data = []
    for _ in range(n):
        age = random.randint(20, 65)
        income = round(random.uniform(2.5, 15.0),2)
        credit_score = random.randint(300, 800)
        has_default = random.choice([0, 1])
        approved = 1 if credit_score > 650 and income > 5 and not has_default else 0
        data.append([age, income, credit_score, has_default, approved])
    return pd.DataFrame(data,columns=['age', 'income', 'credit_score', 'has_default', 'approved'])

df = generateData()

X = df[['age', 'income', 'credit_score', 'has_default']]
y = df['approved']


model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

joblib.dump(model,"decision_Tree_model.pkl")

app = Flask(__name__) #json,restful

model = joblib.load("decision_Tree_model.pkl")

@app.route("/")
def home():
    return "Decision Tree API hazır 🚀"    #http://localhost:5000/

@app.route("/prediction", methods=["POST"]) #http://localhost:5000/prediction  POST
def predict():
    data = request.get_json()
    try: 
        age = int(data["age"])
        income = float(data["income"])
        credit_score = int(data["credit_score"])
        has_default = int(data["has_default"])
        
        testData = np.array([[age, income, credit_score, has_default]])
        result = model.predict(testData)[0]

        return jsonify({
            "age": age,
            "income": income,
            "credit_score": credit_score,
            "has_default": has_default,
            "approved": "Onaylandı" if result == 1 else "Onaylanmadı"
        })

    except Exception as e:
        return jsonify({"hata": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)

    #fast apı ve swagger