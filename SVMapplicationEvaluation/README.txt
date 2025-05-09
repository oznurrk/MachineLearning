Proje Kullanım Rehberi: SVM Başvuru Değerlendirme

Bu proje, sahte (faker) verilerle bir işe başvuru değerlendirme sistemi oluşturur.
Başvuran kişinin tecrübe yılı ve teknik sınav puanı bilgilerine göre,
işe alınır mı, alınmaz mı kararını tahmin eder.

Başlangıç

1. Gerekli Paketler

Projeyi çalıştırmadan önce aşağıdaki Python kütüpanelerinin kurulu olması gerekir:

pip install numpy matplotlib faker scikit-learn joblib

Adım 1: Modeli Eğitmek

İlk olarak SVMApplicationEvaluation.py dosyasını çalıştırıyoruz.

Bu dosyada:

Sahte veri üretimi yapılır.

SVM modeli için en iyi parametreler bulunur.

Model eğitildikten sonra pipeline_model.pkl dosyası olarak kaydedilir.

-Pipeline kullanmamın sebebi eğitilmiş veri ve ölçeklendirdiğimiz veriyi ayrı ayrı dosyalarda kaydedip işlem yapmak istemediğim için; tek dosyada birleştirip işleme devam etmek istememdi.-

Modelin karar sınırları görselleştirilir.

Terminal üzerinden elle veri girerek anlık tahmin yapabilirsin.

Çalıştırmak için:

python SVMApplicationEvaluation.py

Çıktıda şunlar görülecek:

En iyi hiperparametreler
      svc__C: 10
      svc__gamma: scale
      svc__kernel: rbf -ödevde linear dediği için kodda 'linear' yazılı bırakılmıştır. Tüm parametreler girilerek içinden en iyi sonuç veren GridSearchCV sayesinde rbf bulunmuştur- 

Eğitim ve test doğruluk skorları

Confusion matrix ve classification raporu

Karar sınırı grafiği

Çalıştırmanın sonunda pipeline_model.pkl adında bir model dosyası oluşacaktır.

Adım 2: Web Arayüzünden Tahmin Yapmak

Modeli eğittikten sonra, basit bir web arayüzü üzerinden tahmin yapmak için app.py dosyasını çalıştırıyoruz.

Çalıştırmak için:
uvicorn main:app --reload 
ya da 
python -m uvicorn app:app --reload

Tarayıcıda şu adrese git:

http://127.0.0.1:8000

Web Arayüz Kullanımı

Arayüzde senden iki bilgi istenir:

Tecrübe yılı (0-10 arasında bir değer)

Teknik sınav puanı (0-100 arasında bir değer)

Bu bilgileri girip işlem butonuna tıklayarak sonuca ulaşabilirsin:

"Aday işe alınır" veya

"Aday işe alınmaz"

Arka planda, pipeline_model.pkl kullanılmaktadır.

Proje Akış Özeti

1. Eğitim (SVMApplicationEvaluation.py çalıştırılır):
   - Veri üret ➔ Eğit ➔ Modeli kaydet ➔ Test et ➔ Grafik çiz ➔ Terminalden tahmin yap.

2. Web (app.py çalıştırılır):
   - Kaydedilmiş modeli yükle ➔ Kullanıcıdan giriş al ➔ Tahmini göster.
