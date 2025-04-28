
# DBSCAN Ödev Projesi

Bu proje, DBSCAN algoritması kullanılarak veri kümeleri üzerinde kümeleme (clustering) işlemi yapılmasını hedeflemektedir.  
Çalışma Python dili ve ilgili veri bilimi kütüphaneleri kullanılarak geliştirilmiştir.

## Proje Yapısı

- `app/` : Ana uygulama klasörü (veri işleme ve model dosyaları içerir)
- `countries_cluster_plot.png` : Ülkeler veri setinin kümeleme görselleştirmesi
- `products_cluster_plot.png` : Ürünler veri setinin kümeleme görselleştirmesi
- `.env` : Ortam değişkenleri dosyası
- `.venv/` : Sanal ortam klasörü
- `requirements.txt` : Gerekli kütüphaneler listesi

## Kullanılan Teknolojiler

- Python 3.x
- Scikit-learn
- Matplotlib
- FastAPI
- Uvicorn

## Kurulum

```bash
# Sanal ortamı oluşturun ve aktif edin
python -m venv .venv
source .venv/bin/activate  # Windows için: .venv\Scripts\activate

# Gerekli paketleri yükleyin
pip install -r requirements.txt

# Uygulamayı çalıştırın
uvicorn app.main:app --reload
```

## Proje Notları

- DBSCAN algoritması için `eps` ve `min_samples` optimizasyonu yapılmıştır.
- `Kneedle` yöntemi kullanılarak ideal epsilon (`eps`) değeri seçilmiştir.
- Kümeleme sonuçları görselleştirilmiş ve `.png` olarak kaydedilmiştir.

## Katılımcılar

- **Gözde Dilaver**
- **Öznur SAK**
- **Meryemnur Pala**
- **Kübra Tüysüz**
- **Nilgün Demirkaya**