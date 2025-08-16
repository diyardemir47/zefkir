# api.py (GÜNCEL)
import os
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, jsonify, current_app
import traceback
import gc
from sklearn.cluster import MiniBatchKMeans
import base64
from io import BytesIO

# Debug flag from env
DEBUG = os.environ.get('DEBUG', 'False').lower() in ('1', 'true', 'yes')

# -------------------------------------------------------------------
# 0. App config ve limitler
# -------------------------------------------------------------------
MAX_UPLOAD_MB = int(os.environ.get('MAX_UPLOAD_MB', 2))  # default 2 MB
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024

# -------------------------------------------------------------------
# 1. YÜZ ANALİZİ FONKSİYONLARI (SENİN ORJİNAL KODUNUN KORUNMUŞ HALİ)
# -------------------------------------------------------------------

def detect_skin_tone(image, landmarks):
    try:
        points = [117, 123, 143, 147]
        avg_bgr_points = []
        h, w = image.shape[:2]
        for p_idx in points:
            if p_idx >= len(landmarks): continue
            x, y = int(landmarks[p_idx].x * w), int(landmarks[p_idx].y * h)
            roi = image[max(0, y-5):y+5, max(0, x-5):x+5]
            if roi.size > 0:
                avg_bgr_points.append(np.mean(roi, axis=(0, 1)))
        if not avg_bgr_points: return "Belirsiz", None
        avg_bgr = np.mean(avg_bgr_points, axis=0)
        hsv = cv2.cvtColor(np.uint8([[avg_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        v, s = int(hsv[2]), int(hsv[1])
        if v > 190 and s < 50: tone = "Beyaz"
        elif v > 160 and s < 80: tone = "Açık Buğday"
        elif v > 120: tone = "Buğday"
        elif v > 80: tone = "Kahverengi"
        elif v > 50: tone = "Koyu Siyah"
        else: tone = "Kahverengi-Siyah"
        return tone, avg_bgr
    except Exception as e:
        if DEBUG: print(f"Hata detect_skin_tone: {e}")
        return "Belirsiz", None

def detect_eye_color(image, iris_landmarks, debug=False):
    try:
        if not iris_landmarks or len(iris_landmarks) < 4: return "Belirsiz"
        h_img, w_img = image.shape[:2]
        cx = int(iris_landmarks[0].x * w_img)
        cy = int(iris_landmarks[0].y * h_img)
        r_px = int(np.linalg.norm(np.array([iris_landmarks[1].x, iris_landmarks[1].y]) - np.array([iris_landmarks[3].x, iris_landmarks[3].y])) * h_img * 0.6)
        if r_px <= 3: return "Belirsiz"
        x1, y1 = max(0, cx - r_px), max(0, cy - r_px)
        x2, y2 = min(w_img, cx + r_px), min(h_img, cy + r_px)
        roi = image[y1:y2, x1:x2].copy()
        if roi.size == 0 or roi.shape[0] < 6 or roi.shape[1] < 6: return "Belirsiz"
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        lab2 = cv2.merge((cl, a, b))
        rgb = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        v = hsv[:,:,2]; s = hsv[:,:,1]
        dark_mask = (v < 40) | (s < 20)
        mask = ~dark_mask
        if np.count_nonzero(mask) < (roi.shape[0]*roi.shape[1]*0.05): mask = v > 20
        pixels = rgb.reshape(-1,3); mask_flat = mask.flatten()
        px = pixels[mask_flat]
        if px.size == 0: return "Belirsiz"
        n_clusters = 2 if px.shape[0] < 200 else 3
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256, n_init=10)
        labels = kmeans.fit_predict(px)
        centers = kmeans.cluster_centers_.astype(int)
        counts = np.bincount(labels)
        best_idx = np.argmax(counts)
        color_bgr = centers[best_idx]
        color_arr = np.uint8([[color_bgr]])
        hsv_color = cv2.cvtColor(color_arr, cv2.COLOR_BGR2HSV)[0][0]
        h_val, s_val, v_val = int(hsv_color[0]), int(hsv_color[1]), int(hsv_color[2])
        if debug or DEBUG: print("Eye color center BGR:", color_bgr, "HSV:", (h_val, s_val, v_val), "counts:", counts)
        if v_val < 30: return "Siyah"
        if s_val < 30 and v_val > 160: return "Gri"
        if 90 <= h_val <= 130 and s_val > 40: return "Mavi"
        if 35 <= h_val <= 85 and s_val > 30: return "Yeşil"
        if 10 <= h_val <= 35 and s_val > 30: return "Ela"
        if (h_val < 20 or h_val > 160) and s_val > 20 and v_val > 40: return "Kahverengi"
        if s_val < 40: return "Gri"
        return "Belirsiz"
    except Exception as e:
        if debug or DEBUG: print("detect_eye_color error:", e)
        return "Belirsiz"

def analyze_hair_status(image, landmarks, skin_bgr, debug=False):
    try:
        h, w = image.shape[:2]
        SAMPLE_REGIONS = []
        try:
            cx = int(landmarks[10].x * w)
            cy = int(landmarks[10].y * h) - int(0.08*h)
            size = int(0.12 * w)
            SAMPLE_REGIONS.append((cx - size, cy - size, cx + size, cy + size))
        except: pass
        try:
            lx = int(landmarks[127].x * w); ly = int(landmarks[127].y * h)
            rx = int(landmarks[356].x * w); ry = int(landmarks[356].y * h)
            s = int(0.10 * w)
            SAMPLE_REGIONS.append((lx - s, ly - s, lx + s, ly + s))
            SAMPLE_REGIONS.append((rx - s, ry - s, rx + s, ry + s))
        except: pass
        SAMPLE_REGIONS.append((int(0.35*w), int(0.05*h), int(0.65*w), int(0.18*h)))
        pixels = []
        for (x1,y1,x2,y2) in SAMPLE_REGIONS:
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            region = image[y1c:y2c, x1c:x2c]
            if region.size == 0: continue
            region_small = cv2.resize(region, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            px = region_small.reshape(-1,3)
            pixels.append(px)
        if not pixels: return "Belirsiz"
        all_px = np.vstack(pixels)
        lab_all = cv2.cvtColor(all_px.reshape(1,-1,3).astype(np.uint8), cv2.COLOR_BGR2LAB)[0]
        lab_samples = lab_all.reshape(-1,3)
        keep_mask = np.ones(len(lab_samples), dtype=bool)
        if skin_bgr is not None:
            try:
                skin_lab = cv2.cvtColor(np.uint8([[skin_bgr]]), cv2.COLOR_BGR2LAB)[0][0]
                dists = np.linalg.norm(lab_samples - skin_lab, axis=1)
                keep_mask = keep_mask & (dists > 18)
            except: pass
        filtered = all_px.reshape(-1,3)[keep_mask]
        if filtered.size == 0: filtered = all_px.reshape(-1,3)
        n_clusters = 3 if filtered.shape[0] > 500 else 2
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=256, n_init=10)
        labels = kmeans.fit_predict(filtered)
        centers = kmeans.cluster_centers_.astype(int)
        counts = np.bincount(labels, minlength=n_clusters)
        center_labs = cv2.cvtColor(centers.reshape(-1,1,3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1,3)
        skin_lab = None
        if skin_bgr is not None:
            try: skin_lab = cv2.cvtColor(np.uint8([[skin_bgr]]), cv2.COLOR_BGR2LAB)[0][0]
            except: skin_lab = None
        scores = []
        for i, c in enumerate(centers):
            score = counts[i]
            if skin_lab is not None:
                dist_skin = np.linalg.norm(center_labs[i] - skin_lab)
                score = score * (dist_skin / (dist_skin + 1.0))
            scores.append(score)
        best_idx = int(np.argmax(scores))
        hair_bgr = centers[best_idx]
        hsv = cv2.cvtColor(np.uint8([[hair_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        lval = center_labs[best_idx][0]
        hval, sval, vval = int(hsv[0]), int(hsv[1]), int(hsv[2])
        if debug or DEBUG: print("Hair centers:", centers, "counts:", counts, "chosen:", hair_bgr, "HSV:", (hval,sval,vval), "L:", lval)
        if vval < 40 or lval < 30: return "Siyah"
        if sval < 30 and lval > 170: return "Gri/Beyaz"
        if (hval < 12 or hval > 160) and sval > 30 and vval > 50: return "Kızıl"
        if 18 < hval < 40 and lval > 150 and sval > 40: return "Sarı"
        if vval < 120 and sval > 30: return "Koyu Kahve"
        if vval >= 120 and sval > 30: return "Açık Kahve"
        if lval < 80: return "Koyu Kahve"
        return "Belirsiz"
    except Exception as e:
        if debug or DEBUG: print("analyze_hair_status error:", e)
        return "Belirsiz"

def detect_hair_texture(image, landmarks, debug=False):
    try:
        h, w = image.shape[:2]
        y1 = max(0, int(landmarks[10].y * h - 20))
        y2 = int(landmarks[152].y * h)
        x1 = max(0, int(landmarks[127].x * w))
        x2 = min(w, int(landmarks[356].x * w))
        roi = image[y1:y2, x1:x2]
        if roi.size < 500:
            return "Belirsiz"
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = laplacian.var()
        if debug or DEBUG:
            print("Laplacian Variance:", lap_var)
        if lap_var > 300:
            return "Kıvırcık"
        elif lap_var > 100:
            return "Dalgalı"
        else:
            return "Düz"
    except Exception as e:
        if DEBUG: print(f"Hata detect_hair_texture: {e}")
        return "Belirsiz"

def detect_nose_shape(landmarks, width, height):
    try:
        p1 = np.array([landmarks[6].x, landmarks[6].y])
        p2 = np.array([landmarks[168].x, landmarks[168].y])
        p3 = np.array([landmarks[4].x, landmarks[4].y])
        d = np.linalg.norm(np.cross(p3 - p1, p1 - p2)) / np.linalg.norm(p3 - p1) if np.linalg.norm(p3-p1) > 0 else 0
        h_nose = (landmarks[2].y - landmarks[6].y)*height
        w_nose = (landmarks[454].x - landmarks[234].x)*width
        if w_nose<=0: return "Belirsiz"
        ratio = h_nose / w_nose
        if d>0.015 * height: return "Kavisli/Çengel"
        elif ratio>1.2: return "İnce/Uzun"
        else: return "Geniş/Kısa"
    except Exception as e:
        if DEBUG: print(f"Hata detect_nose_shape: {e}")
        return "Belirsiz"

def detect_eyebrow_shape(landmarks):
    try:
        ys_left = [landmarks[i].y for i in [70,63,105,66,107]]
        ys_right = [landmarks[i].y for i in [300,293,334,296,336]]
        std_left = np.std(ys_left)
        std_right = np.std(ys_right)
        thickness_left = abs(landmarks[105].y - landmarks[65].y)
        thickness_right = abs(landmarks[334].y - landmarks[295].y)
        std = (std_left + std_right) / 2
        thickness = (thickness_left + thickness_right) / 2
        shape = "Kavisli" if std > 0.005 else "Düz"
        density = "Kalın" if thickness > 0.03 else "İnce"
        return f"{density}, {shape}"
    except Exception as e:
        if DEBUG: print(f"Hata detect_eyebrow_shape: {e}")
        return "Belirsiz"

def detect_eye_shape(landmarks, width, height):
    try:
        top = landmarks[159]; bot = landmarks[145]; left = landmarks[33]; right = landmarks[133]
        w_eye = (right.x-left.x)*width
        h_eye = (bot.y-top.y)*height
        ratio = h_eye/w_eye if w_eye>0 else 0
        if ratio<0.20: return "Çekik"
        elif ratio<0.30: return "Badem"
        else: return "Büyük"
    except Exception as e:
        if DEBUG: print(f"Hata detect_eye_shape: {e}")
        return "Belirsiz"

def detect_lip_thickness(landmarks, height):
    try:
        top_lip_upper = landmarks[13]
        top_lip_lower = landmarks[14]
        upper_lip_thickness = (top_lip_lower.y - top_lip_upper.y) * height
        bottom_lip_upper = landmarks[12]
        bottom_lip_lower = landmarks[15]
        lower_lip_thickness = (bottom_lip_lower.y - bottom_lip_upper.y) * height
        avg_thickness = (upper_lip_thickness + lower_lip_thickness) / 2
        face_height = (landmarks[152].y - landmarks[10].y) * height
        thickness_ratio = avg_thickness / face_height if face_height > 0 else 0
        if DEBUG:
            print("Lip thickness ratio:", thickness_ratio)
        if thickness_ratio > 0.08:
            return "Kalın"
        elif thickness_ratio > 0.05:
            return "Orta kalınlıkta"
        else:
            return "İnce"
    except Exception as e:
        if DEBUG: print(f"Hata detect_lip_thickness: {e}")
        return "Belirsiz"

def detect_face_length_ratio(landmarks, width, height):
    try:
        top_face = landmarks[10]
        bottom_face = landmarks[152]
        left_face = landmarks[234]
        right_face = landmarks[454]
        face_height = (bottom_face.y - top_face.y) * height
        face_width = (right_face.x - left_face.x) * width
        if face_width <= 0: return "Belirsiz"
        ratio = face_height / face_width
        if DEBUG:
            print("Face Length Ratio:", ratio)
        if ratio > 1.4:
            return "Uzun"
        elif ratio > 1.2:
            return "Orta"
        else:
            return "Kısa"
    except Exception as e:
        if DEBUG: print(f"Hata detect_face_length_ratio: {e}")
        return "Belirsiz"

# -------------------------------------------------------------------
# 2. SINIFLANDIRMA FONKSİYONLARI (orijinal korundu)
# -------------------------------------------------------------------

def classify_phenotype(features):
    # categories truncated for brevity in explanation — full mapping kept as in original (kept below)
    categories = {
        'Kafkas': {'kurallar': {
            'Ten': ['Beyaz', 'Açık Buğday', 'Buğday'],
            'Göz': ['Mavi', 'Yeşil', 'Gri', 'Kahverengi', 'Ela'],
            'Burun': ['İnce/Uzun', 'Kavisli/Çengel'],
            'Göz Şekli': ['Badem', 'Büyük'],
            'Saç Şekli': ['Düz', 'Dalgalı']
        }, 'alt_tipler': {
            'Nordik': {'bölgeler': 'İskandinavya, Almanya', 'kurallar': {
                'Saç': ['Sarı', 'Açık Kahve'], 'Göz': ['Mavi','Yeşil','Gri'],
                'Burun': ['İnce/Uzun'], 'Kaş': ['İnce, Kavisli','İnce, Düz'], 'Göz Şekli': ['Badem'], 'Saç Şekli': ['Düz']
            }},
            'Alpinoid': {'bölgeler': 'Alpler, Fransa', 'kurallar': {
                'Saç': ['Kahverengi','Koyu Kahve'], 'Göz': ['Ela','Yeşil','Kahverengi'],
                'Burun': ['Geniş/Kısa'], 'Kaş': ['Kalın, Kavisli','Kalın, Düz'], 'Göz Şekli': ['Büyük']
            }},
            'Dinarid': {'bölgeler': 'Balkanlar, Anadolu', 'kurallar': {
                'Saç': ['Koyu Kahve','Siyah'], 'Göz': ['Kahverengi'],
                'Burun': ['Kavisli/Çengel'], 'Kaş': ['Kalın, Kavisli','Kalın, Düz'], 'Göz Şekli': ['Büyük']
            }},
            'İranid/Turanid': {'bölgeler': 'İran, Anadolu', 'kurallar': {
                'Ten': ['Buğday'], 'Saç': ['Siyah','Koyu Kahve'], 'Göz': ['Kahverengi'],
                'Burun': ['İnce/Uzun'], 'Kaş': ['Kalın, Kavisli','Kalın, Düz'], 'Göz Şekli': ['Badem']
            }},
            'Azeri': {'bölgeler': 'Kafkasya, Azerbaycan', 'kurallar': {
                'Ten': ['Buğday','Açık Buğday'], 'Saç': ['Koyu Kahve','Siyah'],
                'Göz': ['Ela','Kahverengi'], 'Burun': ['İnce/Uzun','Geniş/Kısa'],
                'Kaş': ['Kalın, Düz','Kalın, Kavisli'], 'Göz Şekli': ['Badem'], 'Saç Şekli': ['Düz', 'Dalgalı']
            }},
            'Levant/Arab': {'bölgeler': 'Levant, Irak, Suriye', 'kurallar': {
                'Ten': ['Buğday','Kahverengi'], 'Saç': ['Siyah','Koyu Kahve'],
                'Göz': ['Kahverengi','Koyu Kahverengi'], 'Burun': ['Geniş/Kısa'],
                'Kaş': ['Kalın, Düz','Kalın, Kavisli'], 'Göz Şekli': ['Büyük'], 'Dudak': ['Kalın']
            }}
        }},
        'Asyalı': {'kurallar': {
            'Ten': ['Sarımsı-Beyaz','Açık Buğday'], 'Göz': ['Siyah','Koyu Kahverengi'],
            'Burun': ['Geniş/Kısa'], 'Göz Şekli': ['Çekik'],
            'Saç Şekli': ['Düz']
        }, 'alt_tipler': {
            'Doğu Asyalı': {'bölgeler': 'Çin, Japonya, Kore', 'kurallar': {
                'Ten': ['Sarımsı-Beyaz'],'Saç': ['Siyah'],'Göz': ['Siyah'],
                'Kaş': ['İnce, Düz'],'Göz Şekli': ['Çekik'], 'Saç Şekli': ['Düz']
            }},
            'Türkik-Moğol': {'bölgeler': 'Orta Asya, Moğolistan', 'kurallar': {
                'Ten': ['Açık Buğday'],'Saç': ['Siyah'],'Göz': ['Siyah'],
                'Kaş': ['İnce, Düz','İnce, Kavisli'],'Göz Şekli': ['Çekik'], 'Saç Şekli': ['Düz']
            }}
        }},
        'Afrikalı': {'kurallar': {
            'Ten': ['Koyu Siyah','Kahverengi-Siyah','Açık Siyah'], 'Göz': ['Siyah','Koyu Kahverengi'],
            'Burun': ['Geniş/Kısa'], 'Göz Şekli': ['Büyük'],
            'Saç Şekli': ['Kıvırcık']
        }, 'alt_tipler': {
            'Sudanid': {'bölgeler': 'Batı Afrika', 'kurallar': {
                'Ten': ['Koyu Siyah'],'Burun': ['Geniş/Kısa'],'Kaş': ['Kalın, Düz','Kalın, Kavisli'],'Göz Şekli': ['Büyük'],
                'Saç Şekli': ['Kıvırcık'], 'Dudak': ['Kalın']
            }}
        }}
    }

    def compute_score_for_cat(cat_rules):
        total_keys = len(cat_rules)
        if total_keys == 0: return 0.0, []
        matched = 0
        matched_keys = []
        for key, vals in cat_rules.items():
            val = features.get(key)
            if val is None: continue
            val_norm = val if not isinstance(val, str) else val.strip()
            if val_norm in vals:
                matched += 1
                matched_keys.append(key)
        score = matched / total_keys
        return score, matched_keys

    main_scores = {}; main_matches = {}
    for cat_name, cat in categories.items():
        score, matched_keys = compute_score_for_cat(cat.get('kurallar', {}))
        main_scores[cat_name] = score
        main_matches[cat_name] = matched_keys

    sorted_main = sorted(main_scores.items(), key=lambda x: x[1], reverse=True)
    if not sorted_main:
        return "Belirsiz", "Belirsiz", "", 0.0, 0.0
    best_main, best_score = sorted_main[0]
    second_score = sorted_main[1][1] if len(sorted_main) > 1 else 0.0

    MIN_MAIN_SCORE = 0.40
    MIN_MAIN_MARGIN = 0.10

    if best_score < MIN_MAIN_SCORE or (best_score - second_score) < MIN_MAIN_MARGIN:
        return "Belirsiz", "Belirsiz", "", round(best_score,2), 0.0

    subcats = categories[best_main].get('alt_tipler', {})
    best_sub = "Belirsiz"; sub_region = ""; conf_sub = 0.0
    if subcats:
        sub_scores = {}
        for sname, sdef in subcats.items():
            sc, matched_keys = compute_score_for_cat(sdef.get('kurallar', {}))
            sub_scores[sname] = (sc, matched_keys, sdef.get('bölgeler', ''))
        s_sorted = sorted(sub_scores.items(), key=lambda x: x[1][0], reverse=True)
        if s_sorted:
            s_best_name, (s_best_score, s_matched_keys, s_region) = s_sorted[0]
            MIN_SUB_SCORE = 0.50
            if s_best_score >= MIN_SUB_SCORE:
                best_sub = s_best_name
                sub_region = s_region
                conf_sub = round(s_best_score,2)

    conf_main = round(best_score,2)
    return best_main, best_sub, sub_region, conf_main, conf_sub

def classify_turkish_phenotype(features):
    turkish_phenotypes = {
        'TÜRK': {
            'puan': 0, 'açıklama': 'Yörük, Tatar, Anadolu Türkmeni, Manav',
            'kurallar': {
                'Yüz Uzunluğu': {'Kısa': 2, 'Orta': 1},
                'Ten': {'Kumral': 2, 'Açık Buğday': 1},
                'Saç': {'Kahverengi': 2},
                'Saç Şekli': {'Düz': 2, 'Dalgalı': 1},
                'Kaş': {'Seyrek, Düz': 2, 'İnce, Düz': 1, 'İnce, Kavisli': 1},
                'Göz Şekli': {'Çekik': 3, 'Çekik/Kısık Badem Göz': 3},
                'Göz': {'Kahverengi': 2},
                'Burun': {'Geniş, kısa ve düz': 2, 'Kısa ve düz olmayan': 1},
            }
        },
        'BALKAN': {
            'puan': 0, 'açıklama': 'Bulgar, Boşnak, Pomak, Makedon, Arnavut, Kosovalı, Balkan Türkmeni',
            'kurallar': {
                'Yüz Uzunluğu': {'Kısa': 1, 'Orta': 1, 'Uzun': 1},
                'Ten': {'Pembemsi Beyaz ve Yandığında Bronzlaşır': 2, 'Porselen': 1, 'Açık Buğday': 1},
                'Saç': {'Sarı': 2, 'Kumral': 1, 'Kahverengi': 1},
                'Saç Şekli': {'Kıvırcık': 1, 'Düz': 1, 'Dalgalı': 1},
                'Kaş': {'İnce, Düz': 1, 'İnce, Kavisli': 1},
                'Göz Şekli': {'Büyük': 1, 'Badem': 1},
                'Göz': {'Mavi': 2, 'Yeşil': 1, 'Ela': 1},
                'Burun': {'Geniş, kısa ve düz': 1, 'Kısa ve düz olmayan': 1, 'Kısa ve düz': 1},
            }
        },
        'KAFKAS': {
            'puan': 0, 'açıklama': 'Çerkes, Abhaz, Abaza, Gürcü, Laz, Ahıska Türkü',
            'kurallar': {
                'Yüz Uzunluğu': {'Uzun': 1, 'Oldukça uzun': 2},
                'Ten': {'Porselen': 2, 'Pembemsi Beyaz ve Yandığında Bronzlaşır': 1},
                'Saç': {'Kumral': 2, 'Kahverengi': 1},
                'Saç Şekli': {'Dalgalı': 1},
                'Kaş': {'İnce, Düz': 1, 'İnce, Kavisli': 1},
                'Göz Şekli': {'Büyük': 1},
                'Göz': {'Yeşil': 2, 'Ela': 2, 'Mavi': 1},
                'Burun': {'Uzun ve düz': 2, 'Uzun ve düz olmayan': 1, 'Kısa ve düz olmayan': 1},
            }
        },
        'İRANİ': {
            'puan': 0, 'açıklama': 'Kürt, Zaza, Fars',
            'kurallar': {
                'Yüz Uzunluğu': {'Orta': 1, 'Uzun': 1},
                'Ten': {'Bronz': 2, 'Açık Buğday': 1, 'Kumral': 1, 'Çikolata': 1},
                'Saç': {'Siyah': 2, 'Kahverengi': 1},
                'Saç Şekli': {'Kıvırcık': 2},
                'Kaş': {'İnce ve Eğri': 2, 'Kalın Düz': 1},
                'Göz Şekli': {'Büyük': 1, 'Gözlerimin kulaklarıma yakın olan ucu aşağı bakıyor': 1},
                'Göz': {'Kahverengi': 1, 'Koyu Kahverengi': 1},
                'Burun': {'Geniş ve düz olmayan': 2, 'Geniş ve Sivri': 1, 'Uzun ve düz olmayan': 1, 'Kısa ve düz olmayan': 1},
            }
        },
        'ORTA DOĞU': {
            'puan': 0, 'açıklama': 'Arap, Berberi, Sefarad Yahudisi',
            'kurallar': {
                'Yüz Uzunluğu': {'Orta': 1, 'Uzun': 1},
                'Ten': {'Çikolata': 1, 'Bronz': 1, 'Açık Buğday': 1},
                'Saç': {'Siyah': 2, 'Kahverengi': 1},
                'Saç Şekli': {'Kıvırcık': 2},
                'Kaş': {'Kalın Düz': 1},
                'Göz Şekli': {'Büyük': 1, 'Gözlerimin kulaklarıma yakın olan ucu aşağı bakıyor': 1},
                'Göz': {'Koyu Kahverengi': 1, 'Kahverengi': 1},
                'Burun': {'Geniş ve Sivri': 2, 'Uzun ve düz olmayan': 1, 'Geniş ve düz olmayan': 1, 'Kısa ve düz olmayan': 1},
                'Dudak': {'Kalın': 2},
            }
        },
        'BATI ANADOLU/HELEN': {
            'puan': 0, 'açıklama': 'Rum, Grek',
            'kurallar': {
                'Yüz Uzunluğu': {'Orta': 2, 'Uzun': 1},
                'Ten': {'Açık Buğday': 1, 'Kumral': 1, 'Porselen': 1},
                'Saç': {'Kumral': 1, 'Kahverengi': 2},
                'Saç Şekli': {'Düz': 1, 'Dalgalı': 1},
                'Kaş': {'İnce, Düz': 1, 'İnce, Kavisli': 1},
                'Göz Şekli': {'Kısık olmayan ve küçük olmayan badem göz': 1},
                'Göz': {'Mavi': 1, 'Yeşil': 1, 'Ela': 1, 'Kahverengi': 1},
                'Burun': {'Geniş, kısa ve düz': 1, 'Uzun ve düz': 1, 'Kısa ve düz': 1, 'Kısa ve düz olmayan': 1},
            }
        },
        'DOĞU ANADOLU/ERMENİ/AZERİ': {
            'puan': 0, 'açıklama': 'Doğu Anadolu, Ermeni, Azeri',
            'kurallar': {
                'Yüz Uzunluğu': {'Kısa': 1, 'Orta': 1, 'Uzun': 1},
                'Ten': {'Çikolata': 1, 'Bronz': 1, 'Açık Buğday': 1},
                'Saç': {'Kahverengi': 1, 'Siyah': 1},
                'Saç Şekli': {'Kıvırcık': 1, 'Düz': 1},
                'Kaş': {'İnce ve Eğri': 2, 'Düz, Orta/İnce kalınlıkta': 1},
                'Göz Şekli': {'Büyük': 1, 'Gözlerimin kulaklarıma yakın olan ucu aşağı bakıyor': 1, 'Kısık olmayan ve küçük olmayan badem göz': 1},
                'Göz': {'Kahverengi': 1, 'Koyu Kahverengi': 1},
                'Burun': {'Uzun ve düz olmayan': 2, 'Kısa ve düz olmayan': 2, 'Geniş ve Sivri': 1},
            }
        }
    }
    analyzed_features = {
        'Yüz Uzunluğu': features.get('Yüz Uzunluğu', 'Belirsiz'),
        'Ten': features.get('Ten', 'Belirsiz'),
        'Saç': features.get('Saç', 'Belirsiz'),
        'Saç Şekli': features.get('Saç Şekli', 'Belirsiz'),
        'Kaş': features.get('Kaş', 'Belirsiz'),
        'Göz Şekli': features.get('Göz Şekli', 'Belirsiz'),
        'Göz': features.get('Göz', 'Belirsiz'),
        'Burun': features.get('Burun', 'Belirsiz'),
        'Dudak': features.get('Dudak', 'Belirsiz')
    }

    scores = {pheno: 0 for pheno in turkish_phenotypes.keys()}
    for pheno_name, pheno_data in turkish_phenotypes.items():
        for feature_name, rules in pheno_data['kurallar'].items():
            feature_value = analyzed_features.get(feature_name)
            if feature_value in rules:
                scores[pheno_name] += rules[feature_value]

    if DEBUG:
        print("Turkish Phenotype Scores:", scores)

    best_phenotype = max(scores, key=scores.get)
    best_score = scores[best_phenotype]
    total_score = sum(scores.values())
    confidence = best_score / total_score if total_score > 0 else 0

    if best_score < 3:
        return "Belirsiz", "Belirsiz", "Belirsiz", 0, 0

    return "Türkiye Fenotipleri", turkish_phenotypes[best_phenotype]['açıklama'], "", best_score, confidence

# -------------------------------------------------------------------
# 3. GLOBAL MEDIAPIPE / HELPER (PERFORMANCE & SAFETY)
# -------------------------------------------------------------------
mpfm = mp.solutions.face_mesh

# create one FaceMesh instance per worker (refine_landmarks False => daha az bellek)
# If you need iris features, consider a separate endpoint to run refine_landmarks selectively.
face_mesh_instance = mpfm.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,  # >>> performans/memory için False önerilir
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def resize_if_needed(img, max_w=900, max_h=900):
    h, w = img.shape[:2]
    if w <= max_w and h <= max_h:
        return img
    scale = min(max_w / float(w), max_h / float(h))
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def build_result_from_image(img):
    """
    img: BGR numpy array (resized if needed)
    returns: tuple (features, generalPhenotype..., turkishPhenotype...)
    """
    h, w = img.shape[:2]
    # face_mesh_instance is global
    res = face_mesh_instance.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res or not getattr(res, 'multi_face_landmarks', None):
        return None, 'Yüz tespit edilemedi'
    lm = res.multi_face_landmarks[0].landmark

    skin, skin_bgr = detect_skin_tone(img, lm)
    face_length_ratio = detect_face_length_ratio(lm, w, h)
    hair_texture = detect_hair_texture(img, lm)
    lip_thickness = detect_lip_thickness(lm, h)

    iris_landmarks = None
    if len(lm) > 477:
        try:
            iris_landmarks = [lm[i] for i in range(473, 478)]
        except Exception:
            iris_landmarks = None

    features = {
        'Yüz Uzunluğu': face_length_ratio,
        'Ten': skin,
        'Göz': detect_eye_color(img, iris_landmarks),
        'Saç': analyze_hair_status(img, lm, skin_bgr),
        'Saç Şekli': hair_texture,
        'Burun': detect_nose_shape(lm, w, h),
        'Kaş': detect_eyebrow_shape(lm),
        'Göz Şekli': detect_eye_shape(lm, w, h),
        'Dudak': lip_thickness
    }

    main, sub, region, conf_main, conf_sub = classify_phenotype(features)
    turkish_main, turkish_sub, turkish_regions, turkish_score, turkish_conf = classify_turkish_phenotype(features)

    # cleanup
    try:
        del res, lm, iris_landmarks
    except: pass
    gc.collect()

    return {
        'features': features,
        'generalPhenotype': {'mainCategory': main, 'subType': sub, 'regions': region, 'confidence': conf_main},
        'turkishPhenotype': {'mainCategory': turkish_main, 'subType': turkish_sub, 'regions': turkish_regions, 'confidence': turkish_conf},
    }, None

# -------------------------------------------------------------------
# 4. FLASK ENDPOINTS
# -------------------------------------------------------------------

@app.route('/analyze', methods=['POST'])
def analyze_image():
    # file presence
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya bulunamadı'}), 400

    f = request.files['file']
    try:
        data = f.read()
        if not data:
            return jsonify({'error': 'Görüntü boş'}), 400
        if len(data) > (MAX_UPLOAD_MB * 1024 * 1024):
            return jsonify({'error': f'Dosya çok büyük. Maksimum {MAX_UPLOAD_MB} MB.'}), 413

        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Görüntü okunamadı'}), 400

        # resize early to reduce mem/time
        img = resize_if_needed(img, max_w=900, max_h=900)

        # process with mediapipe (catch MemoryError separately)
        try:
            result_json, err = build_result_from_image(img)
            if err:
                return jsonify({'error': err}), 400
            return jsonify(result_json), 200
        except MemoryError as me:
            current_app.logger.exception('MemoryError during processing')
            gc.collect()
            return jsonify({'error': 'Sunucu kaynak yetersiz (MemoryError)'}), 503
        except Exception as e:
            current_app.logger.exception('Exception during processing')
            traceback.print_exc()
            gc.collect()
            return jsonify({'error': 'Analiz hatası: ' + str(e)}), 500

    except Exception as e:
        traceback.print_exc()
        current_app.logger.exception('Unhandled exception in analyze_image')
        gc.collect()
        return jsonify({'error': 'Analiz hatası: ' + str(e)}), 500

@app.route('/analyze_base64', methods=['POST'])
def analyze_base64():
    """
    Fallback endpoint: istemci base64 gönderirse kullan.
    Body: {"image_base64": "..."}
    """
    try:
        data = request.get_json(silent=True)
        if not data or 'image_base64' not in data:
            return jsonify({'error': 'image_base64 required'}), 400

        b64 = data['image_base64']
        if not isinstance(b64, str) or len(b64) < 10:
            return jsonify({'error': 'image_base64 invalid'}), 400

        try:
            img_bytes = base64.b64decode(b64)
        except Exception as e:
            return jsonify({'error': 'base64 decode failed: ' + str(e)}), 400

        if len(img_bytes) > (MAX_UPLOAD_MB * 1024 * 1024):
            return jsonify({'error': f'Dosya çok büyük. Maksimum {MAX_UPLOAD_MB} MB.'}), 413

        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Görüntü okunamadı'}), 400

        img = resize_if_needed(img, max_w=900, max_h=900)

        try:
            result_json, err = build_result_from_image(img)
            if err:
                return jsonify({'error': err}), 400
            return jsonify(result_json), 200
        except MemoryError:
            current_app.logger.exception('MemoryError in base64 endpoint')
            gc.collect()
            return jsonify({'error': 'Sunucu kaynak yetersiz (MemoryError)'}), 503
        except Exception as e:
            current_app.logger.exception('Exception in base64 endpoint')
            traceback.print_exc()
            gc.collect()
            return jsonify({'error': 'Analiz hatası: ' + str(e)}), 500

    except Exception as e:
        traceback.print_exc()
        current_app.logger.exception('Unhandled exception in analyze_base64')
        gc.collect()
        return jsonify({'error': 'Analiz hatası: ' + str(e)}), 500

# Health check
@app.route('/', methods=['GET'])
def index():
    return jsonify({'status': 'ok', 'max_upload_mb': MAX_UPLOAD_MB}), 200

# -------------------------------------------------------------------
# 5. RUN (development)
# -------------------------------------------------------------------
if __name__ == '__main__':
    # dev run
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=DEBUG)
