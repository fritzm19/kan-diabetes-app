import streamlit as st
import numpy as np
import torch
import pandas as pd
import plotly.graph_objects as go
import warnings
from config import load_system_prerequisites

# Mengabaikan peringatan spesifik
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, message='.*std().*')

# ==========================================
# 1. INIT & LOAD PREREQUISITES
# ==========================================
st.set_page_config(page_title="Diabetes Risk Classification", layout="wide")

@st.cache_resource
def initialize_app():
    return load_system_prerequisites()

sys = initialize_app()
scaler = sys["scaler"]
model = sys["model"]
splines_data = sys["splines_data"]
feat_importance = sys['feature_importance']
n_features = sys["scaler"].n_features_in_
# df = sys["dataset"]

CLASS_NAMES = [
    "Healthy",
    "Prediabetes",
    "Diabetes"
]

CLASS_COLORS = {
    "Healthy": "#2ecc71",      # hijau
    "Prediabetes": "#f1c40f",  # kuning/goldenrod
    "Diabetes": "#e74c3c"      # merah
}

# --- DICTIONARY UNTUK MAPPING KATEGORI BRFSS 2015 ---
AGE_LABELS = {
    1: "18 - 24 tahun", 2: "25 - 29 tahun", 3: "30 - 34 tahun",
    4: "35 - 39 tahun", 5: "40 - 44 tahun", 6: "45 - 49 tahun",
    7: "50 - 54 tahun", 8: "55 - 59 tahun", 9: "60 - 64 tahun",
    10: "65 - 69 tahun", 11: "70 - 74 tahun", 12: "75 - 79 tahun",
    13: "80 tahun ke atas"
}

EDU_LABELS = {
    1: "Tidak pernah sekolah / TK",
    2: "Pendidikan dasar (setara SD)",
    3: "Pendidikan menengah (belum lulus SMA)",
    4: "Lulus SMA / Paket C / sederajat",
    5: "Pernah kuliah atau pendidikan vokasi (1–3 tahun)",
    6: "Lulus perguruan tinggi (≥ 4 tahun)"
}

INCOME_LABELS = {
    1: "Kurang dari $10,000 (≈ < Rp 133.9 Juta/tahun)",
    2: "$10,000 hingga < $15,000 (≈ Rp 133.9 Juta - Rp 200.9 Juta)",
    3: "$15,000 hingga < $20,000 (≈ Rp 200.9 Juta - Rp 267.8 Juta)",
    4: "$20,000 hingga < $25,000 (≈ Rp 267.8 Juta - Rp 334.8 Juta)",
    5: "$25,000 hingga < $35,000 (≈ Rp 334.8 Juta - Rp 468.7 Juta)",
    6: "$35,000 hingga < $50,000 (≈ Rp 468.7 Juta - Rp 669.6 Juta)",
    7: "$50,000 hingga < $75,000 (≈ Rp 669.6 Juta - Rp 1 Miliar)",
    8: "$75,000 atau lebih (≈ ≥ Rp 1 Miliar/tahun)"
}

# --- KAMUS DESKRIPSI FITUR UNTUK GRAFIK ---
FEATURE_DESCRIPTIONS = {
    "HighBP": "Tekanan Darah Tinggi (Hipertensi)",
    "HighChol": "Riwayat Kolesterol Tinggi",
    "CholCheck": "Pemeriksaan Kolesterol (5 Thn Terakhir)",
    "BMI": "Indeks Massa Tubuh (BMI)",
    "Smoker": "Riwayat Merokok (≥100 batang)",
    "Stroke": "Riwayat Stroke",
    "HeartDiseaseorAttack": "Penyakit Jantung Koroner / Serangan Jantung",
    "PhysActivity": "Aktivitas Fisik / Olahraga",
    "Fruits": "Konsumsi Buah Harian",
    "Veggies": "Konsumsi Sayur Harian",
    "HvyAlcoholConsump": "Konsumsi Alkohol Berlebih",
    "AnyHealthcare": "Kepemilikan Asuransi Kesehatan",
    "NoDocbcCost": "Kendala Biaya Berobat",
    "GenHlth": "Kondisi Kesehatan Umum",
    "MentHlth": "Kesehatan Mental (Hari buruk)",
    "PhysHlth": "Kesehatan Fisik (Hari sakit)",
    "DiffWalk": "Kesulitan Berjalan / Naik Tangga",
    "Sex": "Jenis Kelamin",
    "Age": "Kelompok Usia",
    "Education": "Tingkat Pendidikan",
    "Income": "Skala Pendapatan Tahunan"
}

# ==========================================
# 2. INISIALISASI SESSION STATE
# ==========================================
# Menyimpan posisi halaman saat ini (0 = Landing Page, 1-3 = Form, 4 = Hasil)
if 'step' not in st.session_state:
    st.session_state.step = 0

# Menyimpan jawaban user agar tidak hilang saat ganti halaman
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}

def next_step():
    st.session_state.step += 1

def prev_step():
    st.session_state.step -= 1

def reset_app():
    st.session_state.step = 0
    st.session_state.form_data = {}
    if 'demo_info' in st.session_state:
        del st.session_state['demo_info'] # Hapus memori demo sebelumnya

# ==========================================
# 3. ROUTING HALAMAN (WIZARD)
# ==========================================

# Fungsi pintasan untuk mengambil data 'Golden Sample' dari dataset
# Fungsi pintasan untuk mengambil data acak dari dataset
def load_demo(tipe_demo):
    df = sys["dataset"]
    
    if df is None:
        st.error("Gagal memuat dataset. Pastikan path CSV di config.py sudah benar.")
        return
        
    target_col = 'Diabetes_012' 
    
    GOLDEN_IDX_SEHAT = 5353
    GOLDEN_IDX_PREDIAB = 196769
    GOLDEN_IDX_DIAB = 209799
    GOLDEN_IDX_OVERRIDE = 185703
    
    if tipe_demo == "sehat":
        idx = GOLDEN_IDX_SEHAT
    elif tipe_demo == "prediabetes":
        idx = GOLDEN_IDX_PREDIAB
    elif tipe_demo == "diabetes":
        idx = GOLDEN_IDX_DIAB
    elif tipe_demo == "override":
        idx = GOLDEN_IDX_OVERRIDE
        
    row = df.loc[idx]
    
    # Mapping nama label asli secara otomatis
    label_map = {0.0: "Healthy (0)", 1.0: "Prediabetes (1)", 2.0: "Diabetes (2)"}
    actual_label = label_map.get(row[target_col], "Unknown")

    label_name = actual_label
    
    st.session_state.form_data = {
        'high_bp': int(row['HighBP']), 'high_chol': int(row['HighChol']), 
        'chol_check': int(row['CholCheck']), 'bmi_raw': float(row['BMI']), 
        'smoker': int(row['Smoker']), 'stroke': int(row['Stroke']), 
        'heart_disease': int(row['HeartDiseaseorAttack']), 'phys_activity': int(row['PhysActivity']), 
        'fruits': int(row['Fruits']), 'veggies': int(row['Veggies']), 
        'hvy_alcohol': int(row['HvyAlcoholConsump']), 'healthcare': int(row['AnyHealthcare']), 
        'nodoc_cost': int(row['NoDocbcCost']), 'gen_hlth': int(row['GenHlth']), 
        'ment_hlth': int(row['MentHlth']), 'phys_hlth': int(row['PhysHlth']), 
        'diff_walk': int(row['DiffWalk']), 'sex': int(row['Sex']), 
        'age': int(row['Age']), 'education': int(row['Education']), 'income': int(row['Income'])
    }
    
    st.session_state.demo_info = {
        "index": idx,
        "label": label_name,
        "raw_features": row.drop(target_col).to_dict()
    }
    
    st.session_state.step = 4

# --- HALAMAN 0: LANDING PAGE ---
if st.session_state.step == 0:
    # 1. Injeksi Custom CSS untuk UI Modern & Tombol Warna-Warni
    st.markdown("""
        <style>
        .hero-title {
            font-size: 2.5rem;
            font-weight: 800;
            color: #1E3A8A; /* Biru gelap profesional */
            margin-bottom: 0.5rem;
            line-height: 1.2;
        }
        .hero-subtitle {
            font-size: 1.1rem;
            color: #475569;
            margin-bottom: 2rem;
            font-weight: 400;
        }
        .info-card {
            background-color: #F8FAFC;
            border-left: 5px solid #3B82F6;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            min-height: 140px;
            display: flex;
            flex-direction: column;
        }
        .info-title {
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 0.3rem;
            font-size: 1.05rem;
        }
        .info-text {
            color: #334155;
            font-size: 0.95rem;
            margin: 0;
        }
        .custom-link {
            color: #3B82F6;
            text-decoration: none;
            font-weight: 600;
        }
        .custom-link:hover {
            text-decoration: underline;
        }
        
        /* Modifikasi Warna Background Tombol Mode Cepat */
        /* Menargetkan baris kolom ke-2 (tempat tombol bypass berada) */
        
        /* Tombol Sehat (Hijau Pastel) */
        div[data-testid="stHorizontalBlock"]:nth-of-type(2) div[data-testid="column"]:nth-of-type(1) div[data-testid="stButton"] button {
            background-color: #d1fae5 !important;
            border: 1px solid #10b981 !important;
            color: #065f46 !important;
            font-weight: 600 !important;
        }
        /* Tombol Prediabetes (Kuning Pastel) */
        div[data-testid="stHorizontalBlock"]:nth-of-type(2) div[data-testid="column"]:nth-of-type(2) div[data-testid="stButton"] button {
            background-color: #fef08a !important;
            border: 1px solid #eab308 !important;
            color: #854d0e !important;
            font-weight: 600 !important;
        }
        /* Tombol Diabetes (Merah Pastel) */
        div[data-testid="stHorizontalBlock"]:nth-of-type(2) div[data-testid="column"]:nth-of-type(3) div[data-testid="stButton"] button {
            background-color: #fee2e2 !important;
            border: 1px solid #ef4444 !important;
            color: #991b1b !important;
            font-weight: 600 !important;
        }
        /* Tombol Override/Thresholding (Biru Pastel) */
        div[data-testid="stHorizontalBlock"]:nth-of-type(2) div[data-testid="column"]:nth-of-type(4) div[data-testid="stButton"] button {
            background-color: #dbeafe !important;
            border: 1px solid #3b82f6 !important;
            color: #1e3a8a !important;
            font-weight: 600 !important;
        }
        
        /* Efek Hover untuk ke-4 tombol */
        div[data-testid="stHorizontalBlock"]:nth-of-type(2) div[data-testid="stButton"] button:hover {
            filter: brightness(0.95) !important;
            transform: scale(1.02);
            transition: all 0.2s ease-in-out;
        }
        </style>
    """, unsafe_allow_html=True)

    # 2. Hero Section
    st.markdown('<div class="hero-title">Implementasi Kolmogorov-Arnold Networks (KAN) untuk Klasifikasi Risiko Diabetes dengan Pendekatan Explainable Artificial Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Sistem Pendukung Keputusan Klinis (CDSS) yang transparan, dapat diinterpretasikan (White-Box), dan didasarkan pada data epidemiologi global.</div>', unsafe_allow_html=True)

    # 3. Transparansi Sistem (Feature Cards)
    # col_info1, col_info2, col_info3 = st.columns(3)
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("""
        <div class="info-card">
            <div class="info-title">🧠 Arsitektur Model</div>
            <p class="info-text">
            Menggunakan <a href="https://arxiv.org/abs/2404.19756" target="_blank" class="custom-link">Kolmogorov-Arnold Networks (KAN)</a>
            dengan struktur aditif yang memungkinkan visualisasi pengaruh setiap fitur melalui kurva B-Spline,
            sehingga prediksi model dapat dijelaskan secara lebih transparan.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_info2:
        st.markdown("""
        <div class="info-card">
            <div class="info-title">📊 Basis Data (Ground Truth)</div>
            <p class="info-text">
            Dilatih menggunakan 253.680 rekam medis dari dataset
            <a href="https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset" target="_blank" class="custom-link">BRFSS 2015</a>
            yang dikumpulkan oleh Pusat Pengendalian dan Pencegahan Penyakit Amerika Serikat (CDC) melalui survei kesehatan populasi berskala nasional.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    # with col_info3:
    #     st.markdown("""
    #     <div class="info-card">
    #         <div class="info-title">🎯 Klasifikasi Multikelas</div>
    #         <p class="info-text">Sistem membedah tingkat risiko ke dalam tiga kelas berjenjang (Sehat, Prediabetes, dan Diabetes) untuk memberikan wawasan yang jauh lebih komprehensif.</p>
    #     </div>
    #     """, unsafe_allow_html=True)

    # 4. Expander Detail Metrik & Disclaimer
    with st.expander("📈 Lihat Metrik Performa Model & Disclaimer Klinis"):
        st.markdown("""
        **Metrik Evaluasi Model (Data Uji):**
        * **Balanced Accuracy:** 48.88%
        * **Sensitivitas (Prediabetes & Diabetes):** 95.10% (Telah dikalibrasi via *Thresholding*)
        * **Rata-rata AUC-ROC:** 75.97
        
        **⚠️ Disclaimer Klinis (Penting):**
        Aplikasi ini dikembangkan semata-mata untuk **tujuan penelitian akademis** (Skripsi). Sistem ini merupakan *Clinical Decision Support System* (CDSS) yang dirancang untuk membantu memberikan wawasan berdasarkan data statistik historis, **bukan** untuk menggantikan diagnosis medis profesional. Keputusan medis akhir harus selalu dikonsultasikan dengan dokter atau tenaga kesehatan yang memiliki lisensi.
        """)

    # 5. Call to Action (Area Tombol)
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Mulai Evaluasi Risiko")
    st.write("Silakan pilih metode input data di bawah ini:")
    
    # Tombol Utama (Manual) - Full width
    st.button("📝 Mulai Isi Kuesioner Manual (21 Pertanyaan) ➔", on_click=next_step, type="primary", use_container_width=True)

    st.markdown("<hr style='margin: 1.5em 0;'>", unsafe_allow_html=True)
    
    st.markdown("#### 🧪 Mode Pengujian Cepat (Bypass Kuesioner)")
    st.write("Gunakan profil data sintetis di bawah ini untuk melihat bagaimana model KAN menganalisis dan memberikan penjelasan (XAI) tanpa perlu mengisi form panjang.")
    
    # Tombol Bypass - Sekarang akan otomatis mengikuti CSS warna di atas!
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.button("🟢 Kasus Sehat", on_click=load_demo, args=("sehat",), use_container_width=True)
    with col2:
        st.button("🟡 Kasus Prediabetes", on_click=load_demo, args=("prediabetes",), use_container_width=True)
    with col3:
        st.button("🔴 Kasus Diabetes", on_click=load_demo, args=("diabetes",), use_container_width=True)
    with col4:
        st.button("⚠️ Kasus Thresholding", on_click=load_demo, args=("override",), use_container_width=True)

# --- HALAMAN 1: DEMOGRAFI & GAYA HIDUP DASAR ---
elif st.session_state.step == 1:
    # TRIK UI: Gunakan 3 kolom (Kiri kosong, Tengah untuk Form, Kanan kosong)
    spacer_left, main_col, spacer_right = st.columns([1, 2, 1])
    
    with main_col:
        st.subheader("Tahap 1 dari 3: Demografi & Gaya Hidup Dasar")
        st.progress(33)
        
        fd = st.session_state.form_data
        
        fd['sex'] = st.radio("Jenis Kelamin", options=[0, 1], format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki", index=fd.get('sex', 0))
        fd['age'] = st.selectbox("Kelompok Usia", options=list(AGE_LABELS.keys()), format_func=lambda x: AGE_LABELS[x], index=list(AGE_LABELS.keys()).index(fd.get('age', 5)))
        fd['education'] = st.selectbox(
            "Tingkat Pendidikan",
            options=list(EDU_LABELS.keys()),
            format_func=lambda x: EDU_LABELS[x],
            index=list(EDU_LABELS.keys()).index(fd.get('education', 4)),
            help=
                """
                    Kategori pendidikan diadaptasi dari BRFSS 2015.
                    Tingkat pendidikan Indonesia mungkin tidak sepenuhnya setara dengan kategori asli,
                    namun pilihan berikut digunakan sebagai pendekatan yang paling mendekati.
                """
        )
        fd['income'] = st.selectbox(
            "Skala Pendapatan Tahunan", 
            options=list(INCOME_LABELS.keys()), 
            format_func=lambda x: INCOME_LABELS[x], 
            index=list(INCOME_LABELS.keys()).index(fd.get('income', 5)),
            help="Konversi mata uang menggunakan nilai tukar rata-rata historis tahun 2015 (1 USD = Rp 13.391)."
        )
        
        st.markdown("---")
        fd['smoker'] = st.radio("Apakah Anda pernah menghabiskan setidaknya 100 batang rokok sepanjang hidup Anda?", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya", index=fd.get('smoker', 0))
        fd['fruits'] = st.radio("Apakah Anda mengonsumsi buah setidaknya 1 kali per hari?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak", index=[1, 0].index(fd.get('fruits', 1)))
        fd['veggies'] = st.radio("Apakah Anda mengonsumsi sayuran setidaknya 1 kali per hari?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak", index=[1, 0].index(fd.get('veggies', 1)))

        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            # Fungsi prev_step otomatis menurunkan step dari 1 ke 0 (Landing Page)
            st.button("🏠 Beranda", on_click=prev_step, use_container_width=True)
        with col_btn2:
            st.button("Lanjut ➔", on_click=next_step, type="primary", use_container_width=True)


# --- HALAMAN 2: KONDISI KESEHATAN & AKSES ---
elif st.session_state.step == 2:
    spacer_left, main_col, spacer_right = st.columns([1, 2, 1])
    
    with main_col:
        st.subheader("Tahap 2 dari 3: Kondisi Kesehatan & Akses Medis")
        st.progress(66)
        
        fd = st.session_state.form_data

        fd['gen_hlth'] = st.selectbox(
            "Bagaimana Anda menilai kondisi kesehatan Anda secara umum?",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "1 - Sangat Baik Sekali",
                2: "2 - Sangat Baik",
                3: "3 - Baik",
                4: "4 - Cukup",
                5: "5 - Buruk"
            }[x],
            index=[1, 2, 3, 4, 5].index(fd.get('gen_hlth', 3)),
            help="Penilaian kesehatan secara keseluruhan berdasarkan kondisi fisik dan kesehatan yang Anda rasakan sehari-hari."
        )
        fd['ment_hlth'] = st.number_input("Selama 30 hari terakhir, berapa hari kesehatan mental Anda terasa kurang baik?", min_value=0, max_value=30, value=fd.get('ment_hlth', 0), help="Termasuk stres, kecemasan, depresi, atau masalah emosional lainnya.")
        fd['phys_hlth'] = st.number_input("Selama 30 hari terakhir, berapa hari kesehatan fisik Anda terasa kurang baik?", min_value=0, max_value=30, value=fd.get('phys_hlth', 0), help="Misalnya karena sakit, cedera, atau kondisi kesehatan lainnya.")
        fd['diff_walk'] = st.radio("Apakah Anda mengalami kesulitan serius saat berjalan atau menaiki tangga?", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya", index=fd.get('diff_walk', 0))
        
        st.markdown("---")
        fd['phys_activity'] = st.radio("Apakah Anda melakukan aktivitas fisik atau olahraga dalam 30 hari terakhir (di luar pekerjaan utama)?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak", index=[1, 0].index(fd.get('phys_activity', 1)))
        fd['healthcare'] = st.radio("Apakah Anda memiliki jaminan atau asuransi kesehatan?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak", index=[1, 0].index(fd.get('healthcare', 1)))
        fd['nodoc_cost'] = st.radio("Apakah dalam 12 bulan terakhir Anda pernah tidak berobat ke dokter karena kendala biaya?", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya", index=fd.get('nodoc_cost', 0))
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.button("⬅ Kembali", on_click=prev_step, use_container_width=True)
        with col_btn2:
            st.button("Lanjut ➔", on_click=next_step, type="primary", use_container_width=True)


# --- HALAMAN 3: RIWAYAT MEDIS & BIOMETRIK ---
elif st.session_state.step == 3:
    spacer_left, main_col, spacer_right = st.columns([1, 2, 1])
    
    with main_col:
        st.subheader("Tahap 3 dari 3: Riwayat Medis & Biometrik")
        st.progress(100)
        
        fd = st.session_state.form_data
        
        # -- Kalkulator BMI Cerdas --
        st.markdown("**Indeks Massa Tubuh (BMI)**")
        bmi_mode = st.radio("Pilih metode input:", ["Hitung Otomatis (Tinggi & Berat)", "Input Angka Langsung"], horizontal=True, label_visibility="collapsed")
        
        if bmi_mode == "Hitung Otomatis (Tinggi & Berat)":
            c1, c2 = st.columns(2)
            tinggi = c1.number_input("Tinggi Badan (cm)", min_value=50.0, max_value=300.0, value=165.0, step=1.0)
            berat = c2.number_input("Berat Badan (kg)", min_value=20.0, max_value=300.0, value=65.0, step=1.0)
            
            calculated_bmi = berat / ((tinggi / 100) ** 2)
            st.info(f"**BMI Anda terhitung:** {calculated_bmi:.1f}")
            fd['bmi_raw'] = calculated_bmi
        else:
            fd['bmi_raw'] = st.number_input("Nilai BMI (Body Mass Index)", min_value=10.0, max_value=99.0, value=float(fd.get('bmi_raw', 25.0)), step=0.1)
        
        st.markdown("---")
        fd['high_bp'] = st.radio("Apakah Anda pernah didiagnosis tekanan darah tinggi (hipertensi) oleh tenaga kesehatan?", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya", index=fd.get('high_bp', 0))
        fd['high_chol'] = st.radio("Apakah Anda pernah didiagnosis kolesterol tinggi oleh tenaga kesehatan?", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya", index=fd.get('high_chol', 0))
        fd['chol_check'] = st.radio("Apakah Anda pernah memeriksa kadar kolesterol dalam 5 tahun terakhir?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak", index=[1, 0].index(fd.get('chol_check', 1)))
        fd['stroke'] = st.radio("Apakah Anda pernah mengalami stroke?", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya", index=fd.get('stroke', 0))
        fd['heart_disease'] = st.radio("Apakah Anda memiliki riwayat penyakit jantung koroner atau serangan jantung?", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya", index=fd.get('heart_disease', 0))
        
        # -- Pertanyaan Alkohol Dinamis berdasarkan Jenis Kelamin --
        is_male = fd.get('sex', 0) == 1
        alcohol_text = "Konsumsi alkohol berlebih? (Rata-rata lebih dari 14 minuman beralkohol per minggu)" if is_male else "Konsumsi alkohol berlebih? (Rata-rata lebih dari 7 minuman beralkohol per minggu)"
        
        fd['hvy_alcohol'] = st.radio(alcohol_text, [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya", index=fd.get('hvy_alcohol', 0), help="""
        Definisi mengikuti BRFSS 2015.
        Satu minuman beralkohol setara dengan sekitar satu gelas bir, satu gelas wine, atau satu takaran minuman keras.
        """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.button("⬅ Kembali", on_click=prev_step, use_container_width=True)
        with col_btn2:
            st.button("Analisis Risiko 🚀", on_click=next_step, type="primary", use_container_width=True)

# --- HALAMAN 4: PROSES INFERENSI & HASIL ---
elif st.session_state.step == 4:
    st.title("Hasil Analisis Risiko Diabetes")
    
    # --- TAMBAHKAN BLOK INI: Notifikasi Data Sintetis (Jika pakai mode demo) ---
    if 'demo_info' in st.session_state:
        di = st.session_state.demo_info
        with st.expander("ℹ️ INFORMASI DATASET AKTUAL (GROUND TRUTH)", expanded=True):
            st.markdown(f"Data profil ini ditarik secara acak dari dataset asli (`BRFSS 2015`).")
            st.markdown(f"**Indeks Baris Data:** `{di['index']}`")
            st.markdown(f"**Label Asli dari Dataset:** `{di['label']}`")
            st.markdown("**Isi Nilai Fitur (Input):**")
            
            # Tampilkan data secara horizontal agar ringkas menggunakan st.dataframe
            df_display = pd.DataFrame([di['raw_features']])
            st.dataframe(df_display, hide_index=True)
            
    fd = st.session_state.form_data
    
    # Eksekusi Winsorization untuk BMI
    bmi_clipped = min(fd['bmi_raw'], sys["bmi_upper_limit"])
    if fd['bmi_raw'] > sys["bmi_upper_limit"]:
        st.caption(f"*(Catatan Sistem: Nilai BMI ekstrem disesuaikan ke ambang persentil {sys['bmi_upper_limit']} untuk stabilitas model).*")

    # Susun Array Sesuai Urutan Training BRFSS 2015
    feature_names = [
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", 
        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", 
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", 
        "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
    ]
    
    features = np.array([
        fd['high_bp'], fd['high_chol'], fd['chol_check'], bmi_clipped, fd['smoker'], 
        fd['stroke'], fd['heart_disease'], fd['phys_activity'], fd['fruits'], fd['veggies'], 
        fd['hvy_alcohol'], fd['healthcare'], fd['nodoc_cost'], fd['gen_hlth'], fd['ment_hlth'], 
        fd['phys_hlth'], fd['diff_walk'], fd['sex'], fd['age'], fd['education'], fd['income']
    ]).reshape(1, -1)
    
    # A. Transformasi
    user_data_scaled = scaler.transform(features)
    input_tensor = torch.tensor(user_data_scaled, dtype=torch.float32)
    
    # B. Forward Pass KAN (Untuk probabilitas)
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).numpy()[0]
    
    # C. Implementasi Cascade Thresholding
    st.subheader("Diagnosis Akhir")
    
    # Variabel untuk melacak apakah threshold memanipulasi pemenang Argmax murni
    is_threshold_override = False 
    argmax_winner = np.argmax(probabilities)
    
    if probabilities[2] >= sys["t_diabetes"]:
        st.error("🚨 Pasien diklasifikasikan masuk ke dalam kelompok risiko tinggi **DIABETES**.")
        predicted_class_idx = 2
        predicted_class_name = "Diabetes"
        active_threshold = sys["t_diabetes"]
        if argmax_winner != 2: is_threshold_override = True
            
    elif probabilities[1] >= sys["t_prediabetes"]:
        st.warning("⚠️ Pasien diklasifikasikan masuk ke dalam kelompok **PREDIABETES**.")
        predicted_class_idx = 1
        predicted_class_name = "Prediabetes"
        active_threshold = sys["t_prediabetes"]
        if argmax_winner != 1: is_threshold_override = True
            
    else:
        st.success("✅ Pasien diklasifikasikan **SEHAT**.")
        predicted_class_idx = 0
        predicted_class_name = "Healthy"
        
    
    # --------------------------------------------------
    # Tampilkan persentase probabilitas dengan Custom UI Card
    # --------------------------------------------------
    # Fungsi helper untuk merender kotak persentase (Soft Background + Solid Border)
    def create_metric_card(label, value, color_hex):
        bg_color = f"{color_hex}1A" 
        
        return f"""
        <div style="
            background-color: {bg_color};
            border-left: 6px solid {color_hex};
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        ">
            <p style="margin: 0; font-size: 0.9rem; opacity: 0.8; font-weight: 500;">{label}</p>
            <h2 style="margin: 0; color: {color_hex}; font-weight: 800; font-size: 2rem;">{value}</h2>
        </div>
        """

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(create_metric_card("Healthy", f"{probabilities[0]:.2%}", CLASS_COLORS["Healthy"]), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card("Prediabetes", f"{probabilities[1]:.2%}", CLASS_COLORS["Prediabetes"]), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_card("Diabetes", f"{probabilities[2]:.2%}", CLASS_COLORS["Diabetes"]), unsafe_allow_html=True)
    
    # --- DISCLAIMER (Hanya muncul jika Cascade Threshold "membajak" hasil klasifikasi murni) ---
    if is_threshold_override:
        st.info(
            f"💡 **Clinical Thresholding Applied:** Sistem menggunakan pendekatan stratifikasi risiko. "
            f"Walaupun probabilitas tertinggi berada pada kelas **{CLASS_NAMES[argmax_winner]}**, "
            f"kelas **{predicted_class_name}** dipilih karena telah memenuhi ambang risiko klinis "
            f"({active_threshold:.1%}). Pendekatan ini dirancang untuk mendukung skrining dini "
            f"dengan memprioritaskan identifikasi individu yang berpotensi memiliki risiko lebih tinggi."
        )
    
    # ==============================================================
    # EXPLAINABLE AI: NATIVE KAN B-SPLINE (WHITE-BOX)
    # ==============================================================

    # --------------------------------------------------------------
    # 1. Jalankan 1x forward pass dan ekstrak kontribusi lokal pasien
    # --------------------------------------------------------------
    old_save_act = getattr(model, 'save_act', False)
    model.save_act = True

    with torch.no_grad():
        _ = model(input_tensor)

        # Shape: [1, 3, 21]
        local_contributions = (
            model.spline_postacts[0][0, predicted_class_idx, :]
            .cpu()
            .numpy()
        )

    model.save_act = old_save_act

    # --------------------------------------------------------------
    # 2. Ranking kontribusi fitur lokal (Berdasarkan Nilai Asli)
    # --------------------------------------------------------------
    df_local = pd.DataFrame({
        "Feature": feature_names,
        "Contribution": local_contributions
    })

    # Sort berdasarkan Actual Contribution (Tertinggi/Positif -> Terendah/Negatif)
    df_local = df_local.sort_values(by="Contribution", ascending=False).reset_index(drop=True)
    
    # Tambahkan kolom "Peringkat" (1 sampai 21) di paling kiri
    df_local.insert(0, 'Peringkat', range(1, len(df_local) + 1))

    # --------------------------------------------------------------
    # 3. Tampilkan ranking fitur (Top 5 + Sisa 16)
    # --------------------------------------------------------------
    st.subheader("🎯 Faktor Utama yang Memengaruhi Prediksi")

    # Ambil Top 5 dan Sisa 16
    top5_df = df_local.head(5).copy()
    remaining_df = df_local.iloc[5:].copy()

    top5_features = top5_df["Feature"].tolist()
    remaining_features = remaining_df["Feature"].tolist()

    # Ubah nama kolom secara dinamis agar mencantumkan kelas prediksi
    col_name = f"Spline output ({predicted_class_name})"
    top5_df = top5_df.rename(columns={"Contribution": col_name})
    remaining_df = remaining_df.rename(columns={"Contribution": col_name})

    # Tampilkan Top 5 secara langsung
    st.markdown(f"**Top 5 Fitur Paling Berpengaruh Terhadap Prediksi Model**")
    st.dataframe(
        top5_df[['Peringkat', 'Feature', col_name]].style.format({col_name: "{:.4f}"}),
        hide_index=True,
        width="stretch"
    )

    # Sembunyikan 16 lainnya di dalam Expander
    with st.expander("Lihat 16 Fitur Lainnya (Peringkat 6 - 21)"):
        st.dataframe(
            remaining_df[['Peringkat', 'Feature', col_name]].style.format({col_name: "{:.4f}"}),
            hide_index=True,
            width="stretch"
        )

    # ==============================================================
    # PERHITUNGAN MATEMATIS (LOGITS -> SOFTMAX -> THRESHOLD)
    # ==============================================================
    with st.expander("Lihat Proses Prediksi Data Secara Matematis"):
        st.write("Berikut adalah transparansi alur perhitungan dari output mentah jaringan KAN hingga menjadi keputusan klinis final:")
        
        # 1. Menampilkan Logits & Pembuktian Aditif
        st.markdown("**1. Raw Output (Logits)**")
        st.write("Karena jaringan KAN murni beroperasi secara aditif, nilai Logit akhir untuk setiap kelas adalah **mutlak hasil penjumlahan dari 21 nilai *output Spline* (*post-activation*)** dari masing-masing fitur.")
        
        # Siapkan data struk (receipt) untuk penjumlahan
        receipt_data = []
        for i, feat in enumerate(feature_names):
            receipt_data.append({
                "Fitur": feat,
                "Healthy": model.spline_postacts[0][0, 0, i].item(),
                "Prediabetes": model.spline_postacts[0][0, 1, i].item(),
                "Diabetes": model.spline_postacts[0][0, 2, i].item()
            })

        df_receipt = pd.DataFrame(receipt_data)

        # Hitung Total (Sigma)
        sum_0 = df_receipt["Healthy"].sum()
        sum_1 = df_receipt["Prediabetes"].sum()
        sum_2 = df_receipt["Diabetes"].sum()

        # Tambahkan baris Total di paling bawah
        total_row = pd.DataFrame([{
            "Fitur": "Σ TOTAL (Logit Final)",
            "Healthy": sum_0,
            "Prediabetes": sum_1,
            "Diabetes": sum_2
        }])

        df_receipt_full = pd.concat([df_receipt, total_row], ignore_index=True)

        # Fungsi styling agar baris Total tebal dan mencolok
        def highlight_total_row(row):
            if row['Fitur'] == 'Σ TOTAL (Logit Final)':
                return ['background-color: #e6f2ff; font-weight: 800; color: black'] * len(row)
            return [''] * len(row)

        st.dataframe(
            df_receipt_full.style.apply(highlight_total_row, axis=1).format({
                "Healthy": "{:.4f}",
                "Prediabetes": "{:.4f}",
                "Diabetes": "{:.4f}"
            }),
            hide_index=True,
            width="stretch",
            height=810
        )
        
        # Tampilkan persamaan matematika eksplisit untuk kelas yang diprediksi
        pred_class_name_en = CLASS_NAMES[predicted_class_idx]
        
        # Urutkan nilai dari yang terbesar untuk ilustrasi persamaan
        sorted_vals = df_receipt.sort_values(by=pred_class_name_en, ascending=False)[pred_class_name_en].tolist()
        
        # 2. Menampilkan Softmax
        st.markdown("**2. Transformasi Softmax**")
        st.write("Fungsi Softmax diterapkan untuk mengubah nilai Logit menjadi distribusi probabilitas yang jika dijumlahkan bernilai 1 (100%). Rumus: $P_i = \\frac{e^{z_i}}{\\sum e^{z_j}}$")
        
        df_probs = pd.DataFrame({
            "Kelas": CLASS_NAMES,
            "Probabilitas Softmax": [f"{p:.2%}" for p in probabilities]
        })
        st.dataframe(df_probs, hide_index=True, width="stretch")
        
        # 3. Menampilkan Evaluasi Thresholding
        st.markdown("**3. Evaluasi Cascade Thresholding**")
        st.write("Sistem mengevaluasi probabilitas Softmax dari kelas dengan risiko tertinggi terlebih dahulu menggunakan ambang batas (threshold) klinis:")
        
        # Styling rule evaluation
        def rule_status(condition):
            return "✅ Terpenuhi" if condition else "❌ Tidak Terpenuhi"

        rule1_cond = probabilities[2] >= sys["t_diabetes"]
        rule2_cond = probabilities[1] >= sys["t_prediabetes"]
        
        st.markdown(f"""
        * **Rule 1 (Cek Diabetes):** Apakah Probabilitas Diabetes ({probabilities[2]:.2%}) $\\ge$ Threshold ({sys['t_diabetes']:.1%})? **➔ {rule_status(rule1_cond)}**
        """)
        
        if not rule1_cond:
            st.markdown(f"""
            * **Rule 2 (Cek Prediabetes):** Apakah Probabilitas Prediabetes ({probabilities[1]:.2%}) $\\ge$ Threshold ({sys['t_prediabetes']:.1%})? **➔ {rule_status(rule2_cond)}**
            """)
            
            if not rule2_cond:
                st.markdown(f"""
                * **Rule 3 (Default):** Karena kedua rule di atas tidak terpenuhi, pasien otomatis masuk kategori **Healthy**.
                """)
                
        st.info(f"**Kesimpulan:** Berdasarkan hasil evaluasi aturan di atas, sistem menetapkan keputusan final: **{predicted_class_name}**.")

    # --------------------------------------------------------------
    # 4. Penjelasan Native KAN (Visualisasi Spline)
    # --------------------------------------------------------------
    st.subheader("🔍 Native KAN Multi-Class Spline Analysis")

    st.write(
        f"""
    Prediksi akhir model adalah **{predicted_class_name}**.

    Untuk setiap fitur di bawah ini, ditampilkan seluruh kurva B-Spline menuju:
    - Healthy
    - Prediabetes
    - Diabetes

    Titik marker menunjukkan posisi pasien pada masing-masing kurva. 
    Urutan grafik di bawah ini menyesuaikan dengan tabel peringkat di atas (dari pendorong terbesar hingga penahan terbesar).
    """
    )

    layer0_data = splines_data[0]

    # --- FUNGSI HELPER UNTUK MENGGAMBAR SPLINE ---
    def render_feature_spline(feature_name):
        i = feature_names.index(feature_name)
        st.markdown("---")
        
        # Ambil deskripsi cantik dari dictionary
        desc = FEATURE_DESCRIPTIONS.get(feature_name, feature_name)
        
        # Cari peringkat fitur ini dari df_local untuk judul section
        peringkat = df_local.loc[df_local['Feature'] == feature_name, 'Peringkat'].values[0]
        st.markdown(f"### #{peringkat} - {desc}")
        
        fig = go.Figure()

        # 1. Plot ketiga kelas
        for j, class_name in enumerate(CLASS_NAMES):
            x_curve_z = layer0_data['preacts'][:, j, i]
            y_curve = layer0_data['postacts'][:, j, i]

            dummy_inverse = np.zeros((len(x_curve_z), n_features))
            dummy_inverse[:, i] = x_curve_z
            x_curve_actual = scaler.inverse_transform(dummy_inverse)[:, i]

            sort_order = np.argsort(x_curve_actual)
            x_curve_actual = x_curve_actual[sort_order]
            y_curve = y_curve[sort_order]

            fig.add_trace(
                go.Scatter(
                    x=x_curve_actual,
                    y=y_curve,
                    mode='lines',
                    name=class_name,
                    line=dict(color=CLASS_COLORS[class_name], width=4)
                )
            )

        # --------------------------------------------------
        # 2. Garis Vertikal & Titik Intersect Pasien
        # --------------------------------------------------
        patient_actual_x = features[0, i]

        fig.add_vline(
            x=patient_actual_x, 
            line_width=2, 
            line_dash="dash", 
            line_color="black", 
            opacity=0.6
        )

        for j, class_name in enumerate(CLASS_NAMES):
            patient_y = model.spline_postacts[0][0, j, i].cpu().numpy()

            fig.add_trace(
                go.Scatter(
                    x=[patient_actual_x],
                    y=[patient_y],
                    mode='markers', 
                    name=f'{class_name} (Intersect)',
                    marker=dict(
                        color=CLASS_COLORS[class_name],
                        size=10, 
                        symbol='circle',
                        line=dict(color='black', width=1.5)
                    ),
                    showlegend=False
                )
            )

        # 3. Garis netral & Layout
        fig.add_hline(y=0, line_color="gray", opacity=0.5)

        # Ubah judul dan sumbu X agar menggunakan deskripsi bahasa Indonesia
        fig.update_layout(
            title=f"Dinamika Spline: Pengaruh {desc} terhadap Keputusan Model",
            xaxis_title=desc,
            yaxis_title="Output Spline (Kontribusi Logit)",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=50, b=20)
        )

        # 4. Render ke Layar (Grafik Kiri, Tabel Kanan)
        col_chart, col_table = st.columns([3, 1]) 

        with col_chart:
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            patient_scores = []
            for j, class_name in enumerate(CLASS_NAMES):
                score = model.spline_postacts[0][0, j, i].cpu().item()
                patient_scores.append({
                    "Kelas": class_name,
                    "Kontribusi": score
                })
            
            st.write("<br><br><br>", unsafe_allow_html=True) 
            st.markdown(f"**Tabel Kontribusi:**", unsafe_allow_html=True)
            
            df_patient_scores = pd.DataFrame(patient_scores)
            st.dataframe(
                df_patient_scores.style.format({"Kontribusi": "{:.4f}"}),
                hide_index=True,
                use_container_width=True
            )

    # --- EKSEKUSI RENDER GRAFIK ---
    # Memanggil fungsi render untuk 5 fitur teratas
    for feat in top5_features:
        render_feature_spline(feat)

    # Sembunyikan 16 grafik lainnya di dalam Expander
    with st.expander("Tampilkan Kurva B-Spline untuk 16 Fitur Lainnya (Peringkat 6 - 21)"):
        for feat in remaining_features:
            render_feature_spline(feat)

    # ==============================================================
    # TOMBOL RESET
    # ==============================================================
    st.markdown("---")
    st.button("🔄 Ulangi Analisis", on_click=reset_app, type="primary", width="stretch")