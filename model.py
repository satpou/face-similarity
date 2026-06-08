import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import os
import pickle

IMG_SIZE = (100, 100)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ════════════════════════════════════════════════════════════════════

def detect_and_crop(img_gray):
    """Deteksi dan crop area wajah. Coba beberapa setting kalau gagal."""
    # coba deteksi dengan setting normal dulu
    faces = face_cascade.detectMultiScale(
        img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    # kalau gagal, coba setting lebih longgar
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            img_gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
        )
    if len(faces) == 0:
        return img_gray, False  # wajah tidak terdeteksi
    x, y, w, h = faces[0]
    return img_gray[y:y+h, x:x+w], True

def preprocess_array(img_gray):
    """Grayscale → deteksi wajah → crop → resize → normalisasi → flatten."""
    face, detected = detect_and_crop(img_gray)
    face = cv2.resize(face, IMG_SIZE)
    # histogram equalization: membuat pencahayaan lebih seragam
    face = cv2.equalizeHist(face)
    face = face / 255.0
    return face.flatten(), detected

def preprocess_bytes(image_bytes):
    """Preprocessing dari file upload (bytes)."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, False
    return preprocess_array(img)

def preprocess_file(path):
    """Preprocessing dari path file."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, False
    return preprocess_array(img)

# ════════════════════════════════════════════════════════════════════
# DATA LATIH
# ════════════════════════════════════════════════════════════════════

def load_dataset(dataset_path="dataset"):
    """Baca semua gambar dari folder dataset."""
    X = []
    labels = []
    gagal = []
    for person_name in sorted(os.listdir(dataset_path)):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue
        for filename in os.listdir(person_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(person_folder, filename)
                vec, detected = preprocess_file(path)
                if vec is not None:
                    X.append(vec)
                    labels.append(person_name)
                    if not detected:
                        gagal.append(f"{person_name}/{filename}")
    return np.array(X), np.array(labels), gagal

def train_pca(X, n_components=50):
    """Latih PCA dari data wajah."""
    k = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=k)
    X_pca = pca.fit_transform(X)
    return pca, X_pca

def save_model(pca, X_pca, labels, path="model.pkl"):
    """Simpan model PCA ke file."""
    with open(path, "wb") as f:
        pickle.dump({"pca": pca, "X_pca": X_pca, "labels": labels}, f)

def load_model_file(path="model.pkl"):
    """Muat model dari file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["pca"], data["X_pca"], data["labels"]

# ════════════════════════════════════════════════════════════════════
# DATA UJI
# ════════════════════════════════════════════════════════════════════

def recognize(image_bytes, pca, X_pca, labels, threshold=0.80):
    """
    Kenali wajah dari database.
    Mengembalikan nama, skor, dan status wajah terdeteksi atau tidak.
    """
    vec, detected = preprocess_bytes(image_bytes)
    if vec is None:
        return "Gagal membaca gambar", 0.0, False

    vec_pca = pca.transform(vec.reshape(1, -1))
    similarities = cosine_similarity(vec_pca, X_pca)[0]

    best_idx = np.argmax(similarities)
    best_score = float(similarities[best_idx])
    best_label = labels[best_idx]

    # ambil top 3 kandidat untuk ditampilkan
    top3_idx = np.argsort(similarities)[::-1][:3]
    top3 = [(labels[i], float(similarities[i])) for i in top3_idx]

    if best_score >= threshold:
        return best_label, best_score, detected, top3
    else:
        return "Tidak dikenal", best_score, detected, top3

def compare_two(img1_bytes, img2_bytes, pca, threshold=0.80):
    """Bandingkan dua wajah langsung."""
    v1, detected1 = preprocess_bytes(img1_bytes)
    v2, detected2 = preprocess_bytes(img2_bytes)
    if v1 is None or v2 is None:
        return 0.0, "Gagal membaca gambar", False, False

    p1 = pca.transform(v1.reshape(1, -1))
    p2 = pca.transform(v2.reshape(1, -1))
    score = float(cosine_similarity(p1, p2)[0][0])

    hasil = "Mirip" if score >= threshold else "Tidak mirip"
    return score, hasil, detected1, detected2

def compare_then_vs_now(old_bytes, now_bytes, pca, X_pca, labels, threshold=0.70):
    """
    Bandingkan foto masa kecil vs masa sekarang.
    Threshold lebih rendah (0.70) karena wajah bisa berubah seiring waktu.
    Mengembalikan:
    - skor kemiripan antara dua foto
    - tebakan nama dari database (foto lama)
    - tebakan nama dari database (foto baru)
    - apakah kemungkinan orang yang sama
    """
    v_old, det_old = preprocess_bytes(old_bytes)
    v_now, det_now = preprocess_bytes(now_bytes)

    if v_old is None or v_now is None:
        return 0.0, "Gagal", "Gagal", False, False, False

    # proyeksi ke ruang PCA
    p_old = pca.transform(v_old.reshape(1, -1))
    p_now = pca.transform(v_now.reshape(1, -1))

    # skor kemiripan langsung antar dua foto
    score_direct = float(cosine_similarity(p_old, p_now)[0][0])

    # cari siapa di database yang paling mirip dengan foto lama
    sim_old = cosine_similarity(p_old, X_pca)[0]
    best_old_idx = np.argmax(sim_old)
    best_old_label = labels[best_old_idx]
    best_old_score = float(sim_old[best_old_idx])

    # cari siapa di database yang paling mirip dengan foto baru
    sim_now = cosine_similarity(p_now, X_pca)[0]
    best_now_idx = np.argmax(sim_now)
    best_now_label = labels[best_now_idx]
    best_now_score = float(sim_now[best_now_idx])

    # keputusan: kemungkinan orang sama kalau:
    # 1. skor langsung >= threshold, ATAU
    # 2. keduanya dikenali sebagai orang yang sama di database
    orang_sama = (
        score_direct >= threshold or
        (best_old_label == best_now_label and
         best_old_score >= threshold and
         best_now_score >= threshold)
    )

    return (
        score_direct,
        best_old_label, best_old_score,
        best_now_label, best_now_score,
        det_old, det_now,
        orang_sama
    )