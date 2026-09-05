# Face Emotion Tracking System

Webcam üzerinden gerçek zamanlı yüz ifadesi analiz ederek **Happy / Neutral / Sad / Surprised**
duygularını tahmin eden, kullanıcının kendi yüzüne kalibre olan bir Python uygulaması.

MediaPipe Face Mesh'in 468 landmark noktasından geometrik özellikler (göz açıklığı,
ağız açıklığı, gülümseme indeksi, kaş kalkması) çıkarılır; herhangi bir eğitim
verisi ya da hazır model kullanılmadan, tamamen açıklanabilir bir **kural tabanlı**
algoritma ile duygu tahmini yapılır.

> **Demo:** Uygulamayi calistirip `s` tusuyla ekran goruntusu alarak ya da ekran
> kaydi yaparak `assets/demo.gif` dosyasini ekleyebilir, bu bolume
> `![demo](assets/demo.gif)` seklinde referans verebilirsin.

## Özellikler

- **Kişiye özel kalibrasyon** — ilk kullanımda nötr yüzünü baseline olarak alır, sonraki açılışlarda kayıtlı profili yükler
- **Kural tabanlı, açıklanabilir algoritma** — kara kutu bir model yok, her karar geometrik bir ölçüme dayanır
- Canlı HUD: duygu etiketi, 4 duygu için olasılık bar'ları, son 10 saniyenin duygu zaman çizelgesi, FPS sayacı
- Landmark mesh görselleştirme (açılabilir/kapanabilir)
- Tek tuşla ekran görüntüsü alma
- Webcam/MediaPipe gerektirmeyen, `pytest` ile çalışan birim testleri

## Kurulum

Önce depoyu indirip proje dizinine geçin:

```bash
git clone https://github.com/omergungor11/face-emotion-tracking-system.git
cd face-emotion-tracking-system
```

Ardından sanal ortamı oluşturup bağımlılıkları yükleyin. Aşağıdaki etkinleştirme
komutu macOS/Linux içindir; Windows PowerShell'de `.venv\Scripts\Activate.ps1`,
Komut İstemi'nde (cmd) `.venv\Scripts\activate.bat` kullanın.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Kullanım

```bash
python main.py
```

Seçilen kalibrasyon dosyası henüz yoksa uygulama açılışta kalibrasyon başlatır.
Ekrandaki geri sayım boyunca nötr bir yüz ifadesi takının — bu senin kişisel
referansın olacak. Kalibrasyon bittikten sonra profil kaydedilir ve canlı duygu
tahmini başlar. Varsayılan dosya `calibration_profiles/default.json` konumundadır.

Kayıtlı bir profil varsa sonraki açılışlarda doğrudan duygu tahmini başlar.
Kullanıcı veya çekim koşulları değiştiğinde uygulama penceresi odaktayken `c`
tuşuna basarak yeniden kalibrasyon yapın; yeni ölçüm seçili profil dosyasının
üzerine kaydedilir.

**Tuşlar:**

| Tuş | İşlev |
|-----|-------|
| `c` | Yeniden kalibrasyon başlat |
| `m` | Landmark mesh görünümünü aç/kapat |
| `s` | Ekran görüntüsü kaydet (`assets/screenshots/`) |
| `q` / `ESC` | Çıkış |

`s` ile kaydedilen ekran görüntüleri `assets/screenshots/` dizinine yazılır.
Bu dizindeki görüntüler `.gitignore` kapsamındadır; bu nedenle `git status`
çıktısında görünmezler. Depoda paylaşmak istediğiniz bir demo görüntüsünü
seçip `assets/demo.png` gibi bu dizinin dışındaki bir konuma kopyalayın ve
yalnızca seçtiğiniz dosyayı Git'e ekleyin.

Farklı bir kamera veya kalibrasyon profili kullanmak için:

```bash
python main.py --camera 1 --calibration-file calibration_profiles/omer.json
```

## Nasıl Çalışır

1. **Landmark çıkarımı** — MediaPipe Face Mesh her kareden 468 3D yüz noktası üretir.
2. **Özellik çıkarımı** (`src/features.py`) — göz-arası mesafeyle normalize edilmiş
   4 geometrik özellik hesaplanır: EAR (göz açıklığı), MAR (ağız açıklığı),
   smile index (ağız köşelerinin konumu) ve brow raise (kaş kalkması).
3. **Kalibrasyon** (`src/calibration.py`) — kayıtlı profil yoksa veya `c` tuşuna
   basılırsa birkaç saniye nötr ifade ölçülür; aykırı değerler filtrelenip
   ortalaması kişisel `baseline` olarak seçili profil dosyasına kaydedilir.
4. **Sınıflandırma** (`src/emotion_classifier.py`) — her karede
   `delta = özellikler - baseline` hesaplanır, duygu skorları softmax ile
   olasılığa çevrilir ve son karelerin ortalaması alınarak titreme önlenir.
5. **HUD** (`src/hud.py`) — OpenCV ile sonuçlar canlı olarak çizilir.

## Proje Yapısı

```
face-emotion-tracking-system/
├── main.py                    # giris noktasi
├── src/
│   ├── config.py               # landmark indeksleri, esikler, renkler
│   ├── face_mesh.py            # MediaPipe FaceMesh sarmalayici
│   ├── features.py             # geometrik ozellik cikarimi
│   ├── calibration.py          # baseline yakalama/kaydetme/yukleme
│   ├── emotion_classifier.py   # kural tabanli siniflandirma
│   ├── hud.py                  # OpenCV overlay
│   └── app.py                  # ana dongu
├── tests/                       # webcam gerektirmeyen birim testleri
├── calibration_profiles/        # kisisel baseline dosyalari (gitignore'da)
└── assets/                      # demo gorselleri
```

## Test

```bash
pip install -r requirements-dev.txt
pytest
```

## Sınırlamalar

- Tek yüz takibi için tasarlandı (aynı anda birden fazla kişi desteklenmez).
- Kalibrasyon kişiye özeldir; profil dosyası paylaşılan bir bilgisayarda
  kullanıcılar arasında yeniden yapılmalıdır.
- Aydınlatma koşulları ve kamera açısı geometrik ölçümleri etkileyebilir.

## Lisans

MIT — detaylar için [LICENSE](LICENSE) dosyasına bakın.
