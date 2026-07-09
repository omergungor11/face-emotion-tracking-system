"""Landmark indeksleri ve ayarlanabilir eşik/ağırlık sabitleri.

Landmark indeksleri MediaPipe Face Mesh'in standart 468 noktalı topolojisine
gore secilmistir (goz, kas ve agiz icin yaygin kullanilan referans noktalar).
"""

# --- Landmark indeksleri ---
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (362, 385, 387, 263, 373, 380)

RIGHT_EYEBROW_MID = 105
LEFT_EYEBROW_MID = 334
RIGHT_EYE_TOP = 159
LEFT_EYE_TOP = 386

MOUTH_LEFT_CORNER = 61
MOUTH_RIGHT_CORNER = 291
MOUTH_INNER_TOP = 13
MOUTH_INNER_BOTTOM = 14

# Goz-arasi mesafe icin disi goz koseleri (olcek normalizasyonu)
INTEROCULAR_LEFT = 33
INTEROCULAR_RIGHT = 263

# --- Kalibrasyon ---
CALIBRATION_DURATION_SEC = 3.0
CALIBRATION_MIN_FRAMES = 30

# --- Zamansal yumusatma ---
SMOOTHING_WINDOW = 10

# --- Skorlama agirliklari ---
W_SMILE = 8.0
W_BROW = 6.0
W_MAR = 4.0
W_EAR = 3.0
NEUTRAL_BIAS = 1.0
TEMPERATURE = 1.0

# --- UI ---
EMOTION_COLORS = {
    "Neutral": (180, 180, 180),
    "Happy": (0, 200, 0),
    "Sad": (200, 120, 0),
    "Surprised": (0, 220, 220),
}
