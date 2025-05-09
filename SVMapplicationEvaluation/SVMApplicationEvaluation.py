import numpy as np
import matplotlib.pyplot as plt
from faker import Faker
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import joblib

# Faker veri üretimi
fake = Faker()
np.random.seed(42)

n_samples = 200
tecrube_yili = np.random.uniform(0, 11, n_samples)
teknik_puan = np.random.uniform(0, 101, n_samples)

# Veriyi kurala göre etiketleme
labels = []
for tecrube, teknik in zip(tecrube_yili, teknik_puan):
    if tecrube < 2 and teknik < 60:
        labels.append(1)  # başarısız
    else:
        labels.append(0)  # başarılı

X = np.column_stack((tecrube_yili, teknik_puan))
y = np.array(labels)

# Eğitim ve test ayırımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline oluştur
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC())
])

# Parametre arama için grid tanımla
param_grid = {
    'svc__C': [0.1, 1, 10], #SVM modelindeki C parametresi, hata toleransı ve modelin genelleme kapasitesini belirler.
    'svc__gamma': ['scale', 'auto', 0.01, 0.1, 1], #Gamma parametresi, kernel fonksiyonunun etkisini belirler.
    'svc__kernel': ['linear'] #Hangi kernel fonksiyonunun kullanılacağını belirtir. Örneğin: 'linear','rbf', 'poly', 'sigmoid'
}

# GridSearchCV ile pipeline üzerine arama yap
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print(f"\nEn iyi parametreler: {grid_search.best_params_}")
print(f"En iyi doğruluk skoru (CV): {grid_search.best_score_:.2f}")

# Test verisiyle en iyi modelle tahmin
best_pipeline = grid_search.best_estimator_
y_pred = best_pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.2f}")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

cr = classification_report(y_test, y_pred)
print("\nClassification Report:\n", cr)

# Pipeline modeli kaydet
joblib.dump(best_pipeline, 'pipeline_model.pkl')
print("\nEn iyi pipeline modeli başarıyla kaydedildi: pipeline_model.pkl")

# Veriyi görselleştir
def plot_decision_boundary(pipeline, X, y):
    plt.figure(figsize=(10, 6))

    # X verisini pipeline içindeki scaler ile ölçekle
    X_scaled = pipeline.named_steps['scaler'].transform(X)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='bwr', s=60, edgecolors='k', alpha=0.7)

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = pipeline.named_steps['svc'].decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '--'])

    if hasattr(pipeline.named_steps['svc'], 'support_vectors_'):
        ax.scatter(pipeline.named_steps['svc'].support_vectors_[:, 0],
                   pipeline.named_steps['svc'].support_vectors_[:, 1],
                   s=150, linewidth=1.5, facecolors='none', edgecolors='k')

    plt.title("Pipeline - Faker Verisiyle SVM: Başvuru Değerlendirmesi")
    plt.xlabel("Tecrübe Yılı (standardize)")
    plt.ylabel("Teknik Puan (standardize)")
    plt.grid(True)
    plt.show()

plot_decision_boundary(best_pipeline, X, y)

# Kullanıcıdan veri alarak tahmin yapma
def predict_user(pipeline):
    try:
        tecrube = float(input("Tecrübe yılını girin (0-10): "))
        teknik = float(input("Teknik sınav puanını girin (0-100): "))
        X_new = np.array([[tecrube, teknik]])
        prediction = pipeline.predict(X_new)

        if prediction[0] == 0:
            print("Tahmin: Aday işe alınır.")
        else:
            print("Tahmin: Aday işe alınmaz.")
    except Exception as e:
        print("Hatalı giriş:", e)

while True:
    predict_user(best_pipeline)
    devam = input("Başka bir tahmin yapmak ister misiniz? (e/h): ")
    if devam.lower() != 'e':
        break