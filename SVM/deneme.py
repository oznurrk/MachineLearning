import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Veri oluşturma
X, y = datasets.make_blobs(n_samples=50, centers=2, random_state=42)

# Veriyi ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Veriyi eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model oluşturma ve eğitme
model = SVC(kernel="linear", C=1.0)
model.fit(X_train, y_train)

# Tahmin ve başarı
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Karar sınırını çizdirme
def plot_decision_boundary(model, X, y):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=60, edgecolors='k', alpha=0.7)

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '--'])

    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
               s=150, linewidth=1.5, facecolors='none', edgecolors='k')

    plt.title("SVM ile Sınıflandırma (Standardize Edilmiş Blobs Verisi)")
    plt.xlabel("Özellik 1 (standardize)")
    plt.ylabel("Özellik 2 (standardize)")
    plt.grid(True)
    plt.show()

plot_decision_boundary(model, X_scaled, y)


def credit_risk_status(income, debpt):
    data = np.array([[income, debpt]])
    data_scaled = scaler.transform(data)
    predict = model.predict(data_scaled)[0]
    #print(f"Standardize Edilmiş Veri: {data_scaled}")
    decision_value = model.decision_function(data_scaled)[0]
    #print(f"Model Karar Fonksiyonu Değeri: {decision_value}")
    if predict == 1:
        return f"Bu kişi RİSKLİ olarak değerlendiriliyor. (Gelir: {income}, Borç Oranı: {debpt})"
    else:
        return f"Bu kişi GÜVENİLİR olarak değerlendiriliyor. (Gelir: {income}, Borç Oranı: {debpt})"

print(credit_risk_status(income=5.5, debpt=75))  
print(credit_risk_status(income=8.0, debpt=40))  #
print(credit_risk_status(income=2.5, debpt=75))  # 