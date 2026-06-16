import json
import joblib
import torch
import pickle
import pandas as pd
from kan import KAN
from pathlib import Path

# ==========================================
# 1. KONSTANTA & PARAMETER STATIS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

# Path ke berbagai asset
SCALER_PATH = (BASE_DIR / "assets" / "standard_scaler.joblib")
WEIGHT_PATH = (BASE_DIR / "models" / "shallow_kan-g5-k3-bs1024-lr0.001.pth")
THRESHOLDS_PATH = (BASE_DIR / "assets" / "optimal_cascade_thresholds.json")
PREPROCESSING_PATH = (BASE_DIR / "assets" / "preprocessing_params.json")
SPLINES_PATH = (BASE_DIR / "splines" / "shallow_kan_splines.pkl")
FEATURE_IMPORTANCE_PATH = (BASE_DIR / "assets" / "feature_importance.json")
DATASET_PATH = (BASE_DIR / "assets" / "brfss2015_cleaned.csv")
TARGET_MODEL = 'Shallow_KAN'

# ==========================================
# 2. FUNGSI EKSTRAKSI RESOURCE
# ==========================================
def load_optimal_thresholds():
    try:
        with open(THRESHOLDS_PATH, 'r') as f:
            data = json.load(f)
            t_diab = data[TARGET_MODEL]["t_diabetes"]
            t_prediab = data[TARGET_MODEL]["t_prediabetes"]
            return t_diab, t_prediab
    except FileNotFoundError:
        print("⚠️ File JSON Threshold tidak ditemukan! Menggunakan nilai default.")
        return 0.35, 0.40

def load_system_prerequisites():
    # 1. Load Scaler
    scaler = joblib.load(SCALER_PATH)
    
    # 2. Load Arsitektur & Bobot Model
    model = KAN(
        width=[21, 3],
        grid=5,
        k=3,
        symbolic_enabled=False,
        seed=42,
        auto_save=False,
        device=torch.device('cpu')
    )
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=torch.device('cpu')))
    model.eval()
    
    # 3. Load Thresholds
    t_diab, t_prediab = load_optimal_thresholds()
    
    # 4. Load Preprocessing Params (BMI Limit)
    with open(PREPROCESSING_PATH, 'r') as f:
        prep_data = json.load(f)
        bmi_limit = prep_data["bmi_upper_limit"]
        
    # 5. Load Data Spline (Pre-acts & Post-acts)
    with open(SPLINES_PATH, 'rb') as f:
        splines_data = pickle.load(f)

    with open(FEATURE_IMPORTANCE_PATH) as f:
        feature_importance = json.load(f)
    
    # Load Dataset untuk keperluan sampling (Demo)
    try:
        df_dataset = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        df_dataset = None
        print("⚠️ Dataset tidak ditemukan. Fitur Random Sample Demo mungkin gagal.")
        
    return {
        "scaler": scaler,
        "model": model, 
        "t_diabetes": t_diab, 
        "t_prediabetes": t_prediab,
        "bmi_upper_limit": bmi_limit,
        "splines_data": splines_data,
        "feature_importance": feature_importance,
        "dataset": df_dataset
    }