# Face Emotion Tracking System — Tasarım Dokümanı

**Tarih:** 2026-07-09

## Amaç

Webcam üzerinden gerçek zamanlı yüz ifadesi analizi yaparak 4 temel duyguyu (Happy,
Neutral, Sad, Surprised) tahmin eden, kullanıcının kendi yüzüne kalibre olan,
GitHub'da paylaşılabilir portfolyo kalitesinde bir Python projesi.

## Yaklaşım

Kural tabanlı (rule-based) landmark geometrisi — eğitim verisi/model indirme
gerektirmez, tamamen açıklanabilir. MediaPipe Face Mesh (468 landmark) +
OpenCV + NumPy.

## Proje Yapısı

```
face-emotion-tracking-system/
├── main.py
├── requirements.txt
├── README.md
├── LICENSE (MIT)
├── .gitignore
├── src/
│   ├── config.py            # landmark indeksleri, eşikler, renkler
│   ├── face_mesh.py         # MediaPipe FaceMesh sarmalayıcı
│   ├── features.py          # landmark -> geometrik özellik (EAR, MAR, smile, brow)
│   ├── calibration.py       # nötr baseline yakalama + JSON kaydet/yükle
│   ├── emotion_classifier.py# kalibrasyona göre kural-tabanlı sınıflandırma
│   ├── hud.py               # OpenCV overlay çizimleri
│   └── app.py               # ana döngü orkestrasyonu
├── calibration_profiles/
├── assets/
└── tests/
    ├── test_features.py
    └── test_emotion_classifier.py
```

## Özellikler ve Normalizasyon

Tüm mesafeler göz-arası mesafeyle (interocular distance) normalize edilir
(kameraya uzaklıktan bağımsız olsun diye):

- **EAR** (Eye Aspect Ratio): göz açıklığı, 6 noktalı standart formül
- **MAR** (Mouth Aspect Ratio): ağız iç yükseklik/genişlik oranı
- **Smile Index**: ağız köşelerinin merkeze göre dikey konumu (yukarı=gülümseme)
- **Brow Raise**: kaş-göz üstü mesafesi

## Kalibrasyon

Uygulama başında ~3 saniye nötr ifade ölçülür, medyan/MAD ile aykırı değerler
filtrelenir, ortalama `baseline` olarak `calibration_profiles/*.json` içine
kaydedilir. `c` tuşu ile yeniden kalibrasyon yapılabilir.

## Sınıflandırma

Her karede `delta = features - baseline`. Duygu skorları:
- Happy: pozitif smile delta
- Sad: negatif smile delta
- Surprised: brow + MAR + EAR delta'larının ağırlıklı ortalaması
- Neutral: sabit bias (skorlar eşiği aşmazsa varsayılan)

Skorlar softmax ile olasılığa çevrilir, son N karenin ortalaması alınarak
titreme (flicker) önlenir.

## UI (OpenCV overlay)

- Kalibrasyon ekranı: geri sayım + ilerleme çubuğu
- Ana ekran: duygu etiketi, 4 olasılık bar'ı, son 10 sn duygu timeline'ı, FPS
- Tuşlar: `c` yeniden kalibrasyon, `m` landmark mesh aç/kapa, `s` screenshot, `q`/`ESC` çıkış
- Hata durumları: kamera açılamazsa net mesaj, yüz kaybolursa donuk/gri gösterim (çökme yok)

## Test Stratejisi

`pytest` ile `src/features.py` ve `src/emotion_classifier.py` saf fonksiyonları
sahte landmark/feature verileriyle test edilir — webcam/MediaPipe gerektirmez.

## Paketleme

`requirements.txt` ile tek komut kurulum, README'de demo GIF + kurulum +
algoritma açıklaması, MIT lisans.
