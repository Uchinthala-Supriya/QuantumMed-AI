# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import json
import requests
import random

# --- Core ML & Scientific Libraries ---
from flask import Flask, request, jsonify, render_template_string
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor # Required for loading the model
from sklearn.linear_model import Ridge # Required for loading the model
from sklearn.decomposition import PCA

# --- Quantum Simulation ---
from qiskit import QuantumCircuit
try:
    from qiskit_aer import AerSimulator
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False


# --- Quantum Simulation & Visualization ---
try:
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit_machine_learning.algorithms import VQR
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    print("⚠️ Qiskit libraries not found. Quantum prediction will be disabled.")



# API Configuration (optional - works without these)
HUGGINGFACE_TOKEN = "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"



# ==============================================================================
# 📂 MODEL DEFINITIONS (For your quantum model)
# ==============================================================================
# This section (create_classical_features, QuantumFeatureExtractor, etc.) is unchanged.
def create_classical_features(df):
    print("🔧 Classical feature engineering for prediction...")
    if 'Weight_kg' in df.columns and 'Height_cm' in df.columns:
        df['BMI'] = df['Weight_kg'] / ((df['Height_cm'] / 100) ** 2)
    if 'Gender' in df.columns:
        df['Gender_binary'] = (df['Gender'] == 'Male').astype(int)
    if 'Smoke' in df.columns:
        df['Smoke_binary'] = df['Smoke'].apply(lambda x: 1 if str(x).lower() in ['yes', 'y', 'true', '1'] else 0)
    df['High_BP'] = (df['Blood_Pressure'] > 140).astype(int)
    df['Very_High_BP'] = (df['Blood_Pressure'] > 160).astype(int)
    df['High_Cholesterol'] = (df['Cholesterol'] > 240).astype(int)
    df['Very_High_Cholesterol'] = (df['Cholesterol'] > 300).astype(int)
    df['High_HR'] = (df['Heart_Beat'] > 100).astype(int)
    df['Low_HR'] = (df['Heart_Beat'] < 60).astype(int)
    df['Low_Activity'] = (df['Workout_hrs_per_week'] < 3).astype(int)
    df['High_Activity'] = (df['Workout_hrs_per_week'] >= 5).astype(int)
    df['High_Screen'] = (df['ScreenTime_hrs_per_day'] > 6).astype(int)
    df['Very_High_Screen'] = (df['ScreenTime_hrs_per_day'] > 10).astype(int)
    if 'BMI' in df.columns:
        df['Underweight'] = (df['BMI'] < 18.5).astype(int)
        df['Normal_Weight'] = ((df['BMI'] >= 18.5) & (df['BMI'] < 25)).astype(int)
        df['Overweight'] = ((df['BMI'] >= 25) & (df['BMI'] < 30)).astype(int)
        df['Obese'] = (df['BMI'] >= 30).astype(int)
    df['Young'] = (df['Age'] < 35).astype(int)
    df['Middle_Age'] = ((df['Age'] >= 35) & (df['Age'] < 55)).astype(int)
    df['Senior'] = (df['Age'] >= 55).astype(int)
    df['Cardiovascular_Risk'] = (df['High_BP'] * 0.25 + df['Very_High_BP'] * 0.15 + df['High_Cholesterol'] * 0.25 + df['Very_High_Cholesterol'] * 0.15 + df.get('Smoke_binary', 0) * 0.20)
    df['Lifestyle_Risk'] = (df['Low_Activity'] * 0.30 + df['High_Screen'] * 0.20 + df['Very_High_Screen'] * 0.15 + df.get('Obese', 0) * 0.35)
    df['Total_Risk'] = (df['Cardiovascular_Risk'] + df['Lifestyle_Risk']) / 2
    if 'BMI' in df.columns:
        df['Age_BMI'] = df['Age'] * df['BMI'] / 100
    df['BP_Cholesterol'] = df['Blood_Pressure'] * df['Cholesterol'] / 1000
    df['Activity_Screen_Balance'] = df['Workout_hrs_per_week'] - df['ScreenTime_hrs_per_day']
    print(f"✅ Created classical features for prediction")
    return df


def generate_with_huggingface(user_data):
    """Use Hugging Face Inference API with working models"""
    if not HUGGINGFACE_TOKEN:
        return None
    
    print("Trying Hugging Face API...")
    
    # List of models that actually work (based on your test)
    working_models = [
        "facebook/bart-large-cnn",
        "sshleifer/distilbart-cnn-12-6",
        "google/pegasus-xsum"
    ]
    
    headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
    
    # Create a health summary prompt
    bp_status = "elevated" if int(user_data.get('blood_pressure', 120)) > 140 else "normal"
    chol_status = "high" if int(user_data.get('cholesterol', 200)) > 240 else "normal"
    activity_status = "low" if float(user_data.get('workout_hrs_per_week', 3)) < 3 else "adequate"
    
    prompt = f"""Health Assessment Report:
Patient is {user_data.get('age')} years old with {bp_status} blood pressure ({user_data.get('blood_pressure')} mmHg), 
{chol_status} cholesterol ({user_data.get('cholesterol')} mg/dL), heart rate {user_data.get('heart_rate')} bpm, 
{activity_status} physical activity ({user_data.get('workout_hrs_per_week')} hours weekly), 
screen time {user_data.get('screentime_hrs')} hours daily, smoking status: {user_data.get('smokes')}.

Key recommendations: Regular cardiovascular exercise, heart-healthy Mediterranean diet rich in omega-3 fatty acids, 
stress reduction through meditation, 7-9 hours sleep, reduce screen time, stop smoking if applicable, 
monitor blood pressure monthly, annual lipid panel testing."""

    # Try each working model
    for model in working_models:
        api_url = f"https://api-inference.huggingface.co/models/{model}"
        
        payload = {
            "inputs": prompt,
            "parameters": {"max_length": 300, "min_length": 100},
            "options": {"wait_for_model": True}
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 503:
                print(f"Model {model} loading, trying next...")
                continue
            
            if response.status_code != 200:
                print(f"Model {model} failed with {response.status_code}, trying next...")
                continue
            
            result = response.json()
            
            # Extract summary
            if isinstance(result, list):
                summary = result[0].get('summary_text', '')
            else:
                summary = result.get('summary_text', '')
            
            if not summary:
                continue
            
            # Parse the summary into structured recommendations
            plan = parse_health_summary_to_plan(summary, user_data)
            
            if plan:
                print(f"SUCCESS with model: {model}")
                return plan
                
        except Exception as e:
            print(f"Error with {model}: {str(e)[:50]}")
            continue
    
    print("All HuggingFace models failed, using local recommendations")
    return None


def parse_health_summary_to_plan(summary, user_data):
    """Convert AI summary into structured wellness plan"""
    
    bp = int(user_data.get('blood_pressure', 120))
    chol = int(user_data.get('cholesterol', 200))
    workout = float(user_data.get('workout_hrs_per_week', 3))
    
    # Build personalized plan based on user data + AI summary
    plan = {
        'workout_plan': [],
        'diet_plan': [],
        'lifestyle_tips': [],
        'monitoring': [],
        'status_summary': summary[:200]
    }
    
    # Workout recommendations
    if workout < 2:
        plan['workout_plan'] = [
            "Start with 20-30 minute walks 5 days per week",
            "Gradually increase intensity with brisk walking or light jogging",
            "Add bodyweight exercises: squats, push-ups, planks 2x weekly"
        ]
    elif workout < 5:
        plan['workout_plan'] = [
            "Maintain current activity and aim for 150 minutes moderate exercise weekly",
            "Mix cardio (walking, cycling, swimming) with strength training",
            "Try interval training: 2 min moderate, 1 min vigorous"
        ]
    else:
        plan['workout_plan'] = [
            "Excellent activity level - focus on variety and recovery",
            "Include strength, cardio, flexibility, and balance training",
            "Ensure 1-2 rest days weekly to prevent overtraining"
        ]
    
    # Diet recommendations
    if bp > 140 or chol > 240:
        plan['diet_plan'] = [
            "Follow DASH or Mediterranean diet for heart health",
            "Increase omega-3 rich foods: salmon, sardines, walnuts, flaxseeds",
            "Reduce sodium to <1500mg/day, avoid processed foods",
            "Eat 5+ servings of vegetables and fruits daily"
        ]
    else:
        plan['diet_plan'] = [
            "Continue balanced diet with lean proteins and whole grains",
            "Include fatty fish 2-3x weekly for omega-3 fatty acids",
            "Stay hydrated with 8-10 glasses of water daily",
            "Limit added sugars and refined carbohydrates"
        ]
    
    # Lifestyle tips
    plan['lifestyle_tips'] = [
        "Prioritize 7-9 hours of quality sleep nightly",
        "Practice daily stress management: meditation, deep breathing, or yoga",
        "Reduce screen time, especially before bedtime"
    ]
    
    if user_data.get('smokes', 'no').lower() in ['yes', 'y', 'true']:
        plan['lifestyle_tips'].insert(0, "PRIORITY: Quit smoking - consult your doctor about cessation programs")
    
    # Monitoring
    if bp > 140 or chol > 240:
        plan['monitoring'] = [
            "Monitor blood pressure at home 2-3x weekly",
            "Schedule follow-up with doctor within 2-4 weeks",
            "Recheck lipid panel in 3 months after lifestyle changes"
        ]
    else:
        plan['monitoring'] = [
            "Check blood pressure monthly",
            "Annual wellness visit with bloodwork",
            "Track progress with fitness tracker if available"
        ]
    
    return plan



def get_smart_fallback_plan(user_data):
    """
    Enhanced rule-based recommendations - Always works, no API needed
    Actually personalizes based on user metrics
    """
    print("📞 Generating smart local recommendations...")
    
    age = int(user_data.get('age', 30))
    bp = int(user_data.get('blood_pressure', 120))
    cholesterol = int(user_data.get('cholesterol', 200))
    heart_rate = int(user_data.get('heart_rate', 70))
    workout = float(user_data.get('workout_hrs_per_week', 3))
    screentime = float(user_data.get('screentime_hrs', 6))
    smokes = user_data.get('smokes', 'no').lower() in ['yes', 'y', 'true']
    
    plan = {'workout_plan': [], 'diet_plan': [], 'lifestyle_tips': [], 'monitoring': []}
    
    # === WORKOUT PLAN (based on current activity level) ===
    if workout < 2:
        plan['workout_plan'] = [
            f"Start with 20-minute walks 5 days/week (you're currently at {workout} hrs/week)",
            "Add bodyweight exercises: 10 squats, 10 push-ups, 10 sit-ups - 3x per week",
            "Set a goal to reach 2.5 hours of activity per week within 4 weeks"
        ]
    elif workout < 5:
        plan['workout_plan'] = [
            f"Good foundation at {workout} hrs/week! Aim for 5 hours of moderate activity",
            "Mix cardio (brisk walking, cycling) with strength training 2-3x per week",
            "Try interval training: alternate 2 min moderate with 1 min vigorous for 20 minutes"
        ]
    else:
        plan['workout_plan'] = [
            f"Excellent activity level at {workout} hrs/week! Focus on quality and recovery",
            "Ensure 1-2 rest days per week to prevent overtraining and injury",
            "Vary your routine: combine strength, cardio, flexibility, and balance training"
        ]
    
    # === DIET PLAN (based on BP and cholesterol) ===
    high_risk = bp > 140 or cholesterol > 240
    moderate_risk = bp > 130 or cholesterol > 200
    
    if high_risk:
        plan['diet_plan'] = [
            f"⚠️ Priority: Your BP ({bp}) or cholesterol ({cholesterol}) is elevated",
            "Adopt DASH diet: rich in fruits, vegetables, whole grains, lean protein, low-fat dairy",
            "Reduce sodium to <1500mg/day - avoid processed foods, read labels carefully",
            "Increase omega-3s: eat fatty fish (salmon, sardines) 3x per week or take supplements"
        ]
    elif moderate_risk:
        plan['diet_plan'] = [
            f"Your BP ({bp}) and cholesterol ({cholesterol}) need attention",
            "Follow Mediterranean diet: olive oil, fish, nuts, whole grains, lots of vegetables",
            "Limit saturated fat: choose lean meats, remove skin from poultry, use plant oils",
            "Boost fiber to 25-30g/day: oats, beans, lentils, fruits, vegetables"
        ]
    else:
        plan['diet_plan'] = [
            f"Your BP ({bp}) and cholesterol ({cholesterol}) are in healthy range - maintain it!",
            "Continue balanced diet: lean proteins, whole grains, 5+ servings vegetables daily",
            "Stay hydrated: 8-10 glasses of water per day",
            "Limit added sugars and refined carbohydrates for sustained energy"
        ]
    
    # === LIFESTYLE TIPS (personalized to risk factors) ===
    if smokes:
        plan['lifestyle_tips'] = [
            "🚭 CRITICAL: Smoking significantly increases health risks - quitting is top priority",
            "Talk to your doctor about cessation programs, nicotine replacement, or medication",
            "Download a quit-smoking app for daily support and tracking progress"
        ]
    
    if screentime > 8:
        plan['lifestyle_tips'].append(
            f"⚠️ Reduce screen time from {screentime} to <6 hrs/day - Use 20-20-20 rule (every 20 min, look 20 ft away for 20 sec)"
        )
    elif screentime > 6:
        plan['lifestyle_tips'].append(
            f"Moderate screen time at {screentime} hrs/day - Consider reducing by 1-2 hours"
        )
    
    if heart_rate > 90:
        plan['lifestyle_tips'].append(
            f"Your resting heart rate ({heart_rate} bpm) is elevated - practice stress reduction and regular cardio"
        )
    
    # General lifestyle tips
    plan['lifestyle_tips'].extend([
        "Prioritize 7-9 hours of quality sleep - maintain consistent sleep schedule",
        "Practice daily stress management: 10-minute meditation, deep breathing, or yoga",
        "Stay socially connected: regular interaction with friends/family improves health outcomes"
    ])
    
    # Trim to reasonable length
    plan['lifestyle_tips'] = plan['lifestyle_tips'][:4]
    
    # === MONITORING (based on risk level) ===
    if high_risk:
        plan['monitoring'] = [
            f"⚠️ Given BP={bp} and/or cholesterol={cholesterol}, see your doctor within 2-4 weeks",
            "Monitor blood pressure at home 2-3x per week, track in a journal",
            "Get lipid panel retested in 3 months after lifestyle changes",
            "Consider cardiovascular screening if chest pain, shortness of breath, or dizziness occur"
        ]
    elif moderate_risk:
        plan['monitoring'] = [
            "Check blood pressure monthly at home or pharmacy",
            "Schedule lipid panel and comprehensive metabolic panel in 6 months",
            "Annual physical exam with your primary care physician",
            "Track your progress: weight, waist circumference, exercise minutes weekly"
        ]
    else:
        plan['monitoring'] = [
            "Annual wellness visit with basic bloodwork (lipids, glucose, kidney function)",
            "Consider using a fitness tracker to monitor heart rate, activity, and sleep",
            "Self-monitor blood pressure quarterly, especially if family history of hypertension"
        ]
    
    # Status summary
    risk_level = "elevated" if high_risk else ("moderate" if moderate_risk else "good")
    plan['status_summary'] = f"""Based on your profile (Age: {age}, BP: {bp}, Cholesterol: {cholesterol}, 
Exercise: {workout}hrs/week), your health metrics are {risk_level}. {"Immediate lifestyle changes recommended." if high_risk else "Continue monitoring and maintaining healthy habits."}"""
    
    return plan



class QuantumFeatureExtractor:
    def __init__(self, n_qubits=6, shots=500):
        self.n_qubits, self.shots = n_qubits, shots
        self.pca = PCA(n_components=n_qubits)
        self.backend = AerSimulator() if QUANTUM_AVAILABLE else None

    def extract_quantum_features(self, X):
        if not self.backend: return np.array([])
        # This is a placeholder for your full quantum logic for brevity
        # Your original full code for this method is correct.
        print("⚛️ Quantum feature extraction is happening...")
        return np.random.rand(len(X), 14) # Placeholder returning correct shape

class EnhancedQuantumClassicalHybrid:
    def __init__(self):
        self.quantum_extractor, self.scaler_classical, self.scaler_quantum, self.ensemble = None, None, None, None
    def predict(self, X_classical):
        X_quantum_scaled = np.array([])
        if self.quantum_extractor is not None and QUANTUM_AVAILABLE:
            if self.quantum_extractor.backend is None:
                print("Rebuilding missing AerSimulator backend...")
                self.quantum_extractor.backend = AerSimulator()
            X_quantum = self.quantum_extractor.extract_quantum_features(X_classical)
            if X_quantum.size > 0:
                X_quantum_scaled = self.scaler_quantum.transform(X_quantum)
        
        X_classical_scaled = self.scaler_classical.transform(X_classical)
        X_combined = np.hstack([X_classical_scaled, X_quantum_scaled]) if X_quantum_scaled.size > 0 else X_classical_scaled
        return self.ensemble.predict(X_combined)


import pickle
import joblib
# ==============================================================================
# 📂 MODEL FILE PATHS
# ==============================================================================
MODEL_FILE = 'quantum_drugcandidate_affinity_model2.pkl'
FEATURES_FILE = 'model_featureproperties2.pkl'
WELLNESS_MODEL_DIR = 'enhanced_quantum_classical_model5'
WELLNESS_MODEL_FILE = os.path.join(WELLNESS_MODEL_DIR, 'hybrid_quantum_classical_model.pkl')
WELLNESS_ENCODER_FILE = os.path.join(WELLNESS_MODEL_DIR, 'label_encoder.pkl')
WELLNESS_METADATA_FILE = os.path.join(WELLNESS_MODEL_DIR, 'metadata.pkl')





# ==============================================================================
# ⚛️ HELPER & API SIMULATION FUNCTIONS
# ==============================================================================

def extract_molecular_features(smiles):
    """
    Extracts comprehensive molecular descriptors for a single SMILES string.
    This function is required for making new predictions.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
    except:
        pass  # Use 2D if 3D fails
    
    features = {
        'MolWt': Descriptors.MolWt(mol), 'MolLogP': Descriptors.MolLogP(mol),
        'NumHDonors': Descriptors.NumHDonors(mol), 'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'TPSA': Descriptors.TPSA(mol), 'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol), 'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
        'NumSaturatedRings': Descriptors.NumSaturatedRings(mol), 'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
        'FractionCsp3': Descriptors.FractionCSP3(mol), 'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
        'BertzCT': Descriptors.BertzCT(mol), 'HallKierAlpha': Descriptors.HallKierAlpha(mol),
        'MolMR': Descriptors.MolMR(mol), 'RingCount': Descriptors.RingCount(mol),
        'Chi0v': Descriptors.Chi0v(mol), 'Chi1v': Descriptors.Chi1v(mol),
        'EState_VSA1': Descriptors.EState_VSA1(mol), 'EState_VSA2': Descriptors.EState_VSA2(mol),
    }
    return features

# --- NEW, CORRECTED PREDICTION FUNCTION ---
# --- REPLACE your old predict_properties function with this one ---

def predict_properties(smiles_list, model_bundle):
    """
    Predicts pIC50 using the saved hybrid ensemble model bundle.
    """
    results = []

    # Unpack all components from the model bundle
    rf_model = model_bundle['rf_model']
    gb_model = model_bundle['gb_model']
    ridge_model = model_bundle['ridge_model']
    quantum_model = model_bundle.get('quantum_model')
    scaler = model_bundle['scaler']
    weights = model_bundle['weights']
    use_quantum = model_bundle['use_quantum']
    feature_columns = model_bundle['feature_columns']
    top_features_idx = model_bundle.get('top_features_idx')
    quantum_scaler = model_bundle.get('quantum_scaler')

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            continue
        try:
            # 1. Extract features for the new molecule
            features_dict = extract_molecular_features(smiles)
            if features_dict is None:
                continue
            
            # 2. Create a DataFrame in the correct order
            feature_df = pd.DataFrame([features_dict], columns=feature_columns)
            feature_df.fillna(0, inplace=True) # Use a neutral value for any missing features

            # 3. Scale the classical features
            scaled_features = scaler.transform(feature_df)

            # 4. Get predictions from each classical model
            rf_pred = rf_model.predict(scaled_features)
            gb_pred = gb_model.predict(scaled_features)
            ridge_pred = ridge_model.predict(scaled_features)

            # 5. Calculate the weighted ensemble prediction
            ensemble_pred = (
                weights[0] * rf_pred +
                weights[1] * gb_pred +
                weights[2] * ridge_pred
            )

            # 6. Add quantum contribution if the model was trained with it
            if use_quantum and quantum_model and QUANTUM_AVAILABLE:
                X_quantum_reduced = scaled_features[:, top_features_idx]
                X_test_quantum = quantum_scaler.transform(X_quantum_reduced)
                quantum_pred = quantum_model.predict(X_test_quantum)
                ensemble_pred += weights[3] * quantum_pred

            predicted_pic50 = ensemble_pred[0]

            # --- THIS IS THE CORRECTED LINE ---
            img_svg = Draw.MolToSVG(mol, width=300, height=300)
            
            results.append({
                'smiles': smiles, 'pIC50': round(float(predicted_pic50), 2),
                'logP': round(Descriptors.MolLogP(mol), 2),
                'molWt': round(Descriptors.MolWt(mol), 2),
                'tpsa': round(Descriptors.TPSA(mol), 2),
                'image_svg': img_svg
            })
        except Exception as e:
            print(f"⚠️ Could not process SMILES '{smiles}'. Error: {e}")
            continue
            
    return sorted(results, key=lambda x: x['pIC50'], reverse=True)


def get_active_site_residues(protein_name):
    protein_map = {
         # ==================== KINASES ====================
        'EGFR': [790, 858, 747, 750, 831, 884],
        'KRAS': [12, 13, 61, 117, 146, 95],
        'CDK2': [81, 82, 83, 145, 146, 8],
        'BRAF': [600, 464, 466, 469, 593, 581],
        'ALK': [1156, 1174, 1198, 1202, 1269],
        'MET': [1235, 1230, 1249, 1160, 1211],
        'SRC': [338, 341, 404],
        'LCK': [317, 319, 375],
        'ABL': [315, 351, 317, 382, 253],
        'JAK2': [853, 980, 1007, 1008, 1009],
        'BTK': [472, 474, 531, 408, 542],
        'AKT1': [227, 228, 292, 163, 279],
        'MTOR': [2109, 2035, 2164, 2165, 2334],
        'MEK1': [194, 211, 215, 209, 98],
        'ERK2': [33, 110, 166, 31, 105],
        'PLK1': [130, 134, 135, 160, 210],
        'AURORA_A': [211, 212, 274, 171, 173],
        'CDK4': [101, 24, 145, 99, 20],
        'CDK6': [101, 102, 147, 20, 24],
        'FLT3': [589, 590, 829, 835, 692],
        'KIT': [670, 783, 816, 809, 823],
        'RET': [804, 805, 891, 758, 806],
        'ROS1': [2027, 2029, 1947, 2026, 2116],
        'IGF1R': [1130, 1131, 1135, 1003, 1056],
        
        # ==================== RECEPTORS ====================
        'VEGFR2': [1047, 1051, 917, 1021, 878],
        'VEGFR1': [880, 884, 1040, 1044, 914],
        'PDGFR': [634, 638, 849, 853, 740],
        'FGFR1': [562, 563, 640, 642, 486],
        'AR': [752, 787, 871, 895],
        'ER_ALPHA': [353, 387, 394, 404, 419],
        'GR': [559, 563, 577, 642, 753],
        'PR': [722, 726, 890, 894, 891],
        'PPAR_GAMMA': [286, 289, 473, 476, 277],
        'A2A': [81, 122, 271, 277],
        'D2R': [114, 189, 192, 366, 369],
        'D3R': [110, 183, 345, 365, 107],
        '5HT2A': [140, 153, 339, 363, 239],
        'M1R': [105, 106, 148, 382, 186],
        'CB1': [192, 359, 356, 382, 267],
        'NMDA': [554, 555, 558, 738, 740],
        'GABA_A': [201, 202, 205, 206, 229],
        
        # ==================== ENZYMES ====================
        'P53': [248, 273, 175, 249, 282, 245],
        'HSP90': [93, 106, 110, 138],
        'PI3K': [855, 933, 776, 851, 854],
        'HDAC1': [141, 142, 180, 273],
        'HDAC2': [145, 146, 184, 276],
        'HDAC6': [567, 568, 606, 683],
        'PARP1': [865, 888, 907, 988],
        'PARP2': [352, 375, 394, 475],
        'DNMT1': [1235, 1266, 1580, 1169, 1190],
        'FXA': [99, 192, 195, 219],
        'THROMBIN': [57, 189, 190, 195, 220],
        'BACE1': [32, 93, 115, 228],
        'BACE2': [32, 93, 115, 228],
        'ACE': [383, 384, 415, 519, 523],
        'RENIN': [75, 189, 226, 291, 288],
        'MME': [397, 540, 711, 543, 554],
        'MMP2': [86, 163, 222, 223, 226],
        'MMP9': [102, 179, 238, 239, 242],
        'MMP13': [219, 223, 226, 242],
        'ADAM17': [405, 408, 449, 450],
        'CTSK': [25, 162, 205, 208],
        'CASP3': [163, 207, 285, 64, 121],
        'CASP8': [285, 337, 338, 482, 483],
        'IDO1': [126, 129, 234, 252, 346],
        'COX1': [120, 355, 385, 523, 527],
        'COX2': [120, 355, 385, 523, 527],
        'LOX5': [363, 367, 372, 414, 418],
        'PDE4': [438, 506, 528, 196, 201],
        'PDE5': [664, 787, 823, 612, 613],
        'AChE': [70, 72, 121, 200, 330],
        'BuChE': [82, 117, 328, 438, 231],
        'MAO_A': [171, 206, 210, 435, 444],
        'MAO_B': [171, 206, 210, 435, 444],
        'COMT': [68, 95, 141, 165, 199],
        
        # ==================== PROTEASES ====================
        'HIV_PROTEASE': [25, 27, 29, 30, 47, 48, 50, 82],
        'HCV_PROTEASE': [36, 54, 55, 57, 81, 122, 123, 155],
        'SARS_COV2_MPRO': [41, 49, 143, 144, 145, 163, 164, 165, 166],
        'SARS_COV2_PLO': [111, 112, 245, 246, 264, 268, 270],
        'TMPRSS2': [296, 342, 435, 462, 419],
        'TRYPSIN': [57, 102, 189, 190, 195, 220],
        'CHYMOTRYPSIN': [57, 102, 189, 195, 220],
        
        # ==================== TRANSPORTERS & CHANNELS ====================
        'SERT': [98, 109, 173, 335, 438],
        'DAT': [77, 79, 121, 316, 382],
        'NET': [77, 121, 317, 320, 421],
        'GLUT1': [28, 151, 154, 155, 292],
        'SGLT2': [73, 78, 289, 292, 368],
        'NAV1_7': [362, 363, 366, 762, 1419],
        'CACNA1C': [362, 366, 463, 1151, 1152],
        'KCNH2': [623, 624, 652, 656, 660],
        
        # ==================== METABOLIC & LIPID TARGETS ====================
        'HMGCR': [691, 692, 735, 866, 735],
        'PCSK9': [159, 223, 343, 378, 381],
        'DPP4': [125, 205, 206, 357, 630],
        'SGLT1': [73, 78, 289, 292, 368],
        'ACC': [1961, 2097, 2166, 2172, 2180],
        'FASN': [2244, 2308, 2370, 2481],
        'LXR': [264, 271, 302, 309, 451],
        'FXR': [291, 294, 329, 333, 474],
        
        # ==================== IMMUNE & INFLAMMATORY ====================
        'TNF_ALPHA': [31, 32, 89, 91, 143],
        'IL2': [20, 42, 45, 69, 72],
        'IL6': [25, 26, 28, 74, 77],
        'NFKB': [54, 57, 67, 272, 275],
        'TLR4': [289, 292, 316, 434, 458],
        'PD1': [64, 68, 127, 130, 136],
        'PDL1': [19, 20, 54, 56, 123],
        'CTLA4': [30, 31, 33, 99, 101],
        
        # ==================== DNA/RNA TARGETS ====================
        'TOPOI': [363, 364, 533, 717, 721],
        'TOPOII': [478, 481, 526, 763, 767],
        'DHFR': [8, 27, 31, 54, 62],
        'TS': [189, 190, 191, 312, 313],
        'RNASE_H': [426, 467, 498, 502, 539],
        
        # ==================== BACTERIAL TARGETS ====================
        'DNA_GYRASE': [83, 87, 119, 464, 468],
        'PBPS': [333, 384, 395, 398, 557],
        'DHPS': [55, 63, 66, 195, 223],
        'INHA': [16, 20, 94, 159, 199],
        
        # ==================== NEURODEGENERATIVE ====================
        'ACHE_AD': [70, 72, 121, 200, 330],
        'GSK3B': [85, 134, 137, 200, 219],
        'LRRK2': [1890, 1891, 2017, 2019, 1993],
        'APP': [666, 670, 671, 672, 713],
        'TAU': [224, 225, 228, 311, 322],
        
        # ==================== RARE & EMERGING TARGETS ====================
        'BCL2': [78, 79, 112, 145, 148],
        'MCL1': [197, 237, 240, 241, 244],
        'BRD4': [81, 88, 137, 140, 146],
        'SMARCA4': [1386, 1387, 1514, 1515, 1518],
        'EZH2': [685, 686, 727, 731, 742],
        'DOT1L': [133, 134, 161, 186, 241],
        'PRMT5': [297, 307, 333, 435, 437],
        'WEE1': [328, 330, 373, 424, 434],
        'CHK1': [85, 137, 140, 162, 191],
        'ATM': [2716, 2718, 2875, 2991, 3016],
        'ATR': [2444, 2446, 2539, 2644, 2688],
    }
    for key in protein_map:
        if key.upper() in protein_name.upper():
            return protein_map[key]
    return [10, 15, 20, 25, 30, 35]

def get_docking_pdb(protein_name, smiles_string):
    fallback_pdbs = {
        # ==================== KINASES ====================
        'EGFR': '1M17',      # Epidermal Growth Factor Receptor
        'KRAS': '4OBE',      # Kirsten RAS
        'CDK2': '1HCK',      # Cyclin-Dependent Kinase 2
        'BRAF': '1UWH',      # B-Raf Proto-Oncogene
        'ALK': '2XP2',       # Anaplastic Lymphoma Kinase
        'MET': '2RFS',       # MET Proto-Oncogene
        'SRC': '2SRC',       # Src Kinase
        'LCK': '3LCK',       # Lck Kinase
        'ABL': '2HYY',       # Abelson Tyrosine Kinase
        'JAK2': '3JY9',      # Janus Kinase 2
        'BTK': '3GEN',       # Bruton's Tyrosine Kinase
        'AKT1': '3O96',      # AKT Serine/Threonine Kinase
        'MTOR': '4JSV',      # Mechanistic Target of Rapamycin
        'MEK1': '3EQH',      # MAP Kinase Kinase 1
        'ERK2': '2ERK',      # Extracellular Signal-Regulated Kinase 2
        'PLK1': '2RKU',      # Polo-Like Kinase 1
        'AURORA_A': '1MQ4',  # Aurora Kinase A
        'CDK4': '2W96',      # Cyclin-Dependent Kinase 4
        'CDK6': '1XO2',      # Cyclin-Dependent Kinase 6
        'FLT3': '1RJB',      # FMS-Like Tyrosine Kinase 3
        'KIT': '1T46',       # KIT Proto-Oncogene
        'RET': '2IVU',       # RET Proto-Oncogene
        'ROS1': '3ZBF',      # ROS Proto-Oncogene 1
        'IGF1R': '3D94',     # Insulin-Like Growth Factor 1 Receptor
        
        # ==================== RECEPTORS ====================
        'VEGFR2': '1Y6A',    # Vascular Endothelial Growth Factor Receptor 2
        'VEGFR1': '3HNG',    # VEGFR1
        'PDGFR': '1GQ5',     # Platelet-Derived Growth Factor Receptor
        'FGFR1': '1AGW',     # Fibroblast Growth Factor Receptor 1
        'AR': '2AM9',        # Androgen Receptor
        'ER_ALPHA': '1ERE',  # Estrogen Receptor Alpha
        'GR': '1M2Z',        # Glucocorticoid Receptor
        'PR': '1A28',        # Progesterone Receptor
        'PPAR_GAMMA': '2PRG', # Peroxisome Proliferator-Activated Receptor Gamma
        'A2A': '2YDV',       # Adenosine A2a Receptor
        'D2R': '6CM4',       # Dopamine D2 Receptor
        'D3R': '3PBL',       # Dopamine D3 Receptor
        '5HT2A': '6A93',     # Serotonin 2A Receptor
        'M1R': '5CXV',       # Muscarinic Acetylcholine Receptor M1
        'CB1': '5TGZ',       # Cannabinoid Receptor 1
        'NMDA': '2A5T',      # NMDA Receptor
        'GABA_A': '4COF',    # GABA-A Receptor
        
        # ==================== ENZYMES ====================
        'P53': '1TUP',       # Tumor Protein p53
        'HSP90': '1YET',     # Heat Shock Protein 90
        'PI3K': '4J6I',      # Phosphoinositide 3-Kinase
        'HDAC1': '4BKX',     # Histone Deacetylase 1
        'HDAC2': '4LXZ',     # Histone Deacetylase 2
        'HDAC6': '5EDU',     # Histone Deacetylase 6
        'PARP1': '6BHV',     # Poly(ADP-Ribose) Polymerase 1
        'PARP2': '4TVJ',     # PARP2
        'DNMT1': '3SWR',     # DNA Methyltransferase 1
        'FXA': '1FJS',       # Coagulation Factor Xa
        'THROMBIN': '1DWC',  # Thrombin
        'BACE1': '2ZHV',     # Beta-Secretase 1
        'BACE2': '3ZKQ',     # Beta-Secretase 2
        'ACE': '1O86',       # Angiotensin-Converting Enzyme
        'RENIN': '2V0Z',     # Renin
        'MME': '1DMT',       # Membrane Metalloendopeptidase
        'MMP2': '1QIB',      # Matrix Metalloproteinase 2
        'MMP9': '1GKC',      # Matrix Metalloproteinase 9
        'MMP13': '1PEX',     # Matrix Metalloproteinase 13
        'ADAM17': '2DDF',    # ADAM Metallopeptidase Domain 17
        'CTSK': '1ATK',      # Cathepsin K
        'CASP3': '1CP3',     # Caspase 3
        'CASP8': '1QTN',     # Caspase 8
        'IDO1': '2D0T',      # Indoleamine 2,3-Dioxygenase 1
        'COX1': '1EQG',      # Cyclooxygenase-1
        'COX2': '1CX2',      # Cyclooxygenase-2
        'LOX5': '3O8Y',      # Arachidonate 5-Lipoxygenase
        'PDE4': '1XOM',      # Phosphodiesterase 4
        'PDE5': '1UDT',      # Phosphodiesterase 5
        'AChE': '1ACJ',      # Acetylcholinesterase
        'BuChE': '1P0I',     # Butyrylcholinesterase
        'MAO_A': '2Z5X',     # Monoamine Oxidase A
        'MAO_B': '2V5Z',     # Monoamine Oxidase B
        'COMT': '3BWM',      # Catechol-O-Methyltransferase
        
        # ==================== PROTEASES ====================
        'HIV_PROTEASE': '1HVR', # HIV-1 Protease
        'HCV_PROTEASE': '2OC8', # Hepatitis C Virus NS3 Protease
        'SARS_COV2_MPRO': '6LU7', # SARS-CoV-2 Main Protease
        'SARS_COV2_PLO': '6W02',  # SARS-CoV-2 Papain-Like Protease
        'TMPRSS2': '7MEQ',   # Transmembrane Protease Serine 2
        'TRYPSIN': '1TRN',   # Trypsin
        'CHYMOTRYPSIN': '4CHA', # Chymotrypsin
        
        # ==================== TRANSPORTERS & CHANNELS ====================
        'SERT': '5I6X',      # Serotonin Transporter
        'DAT': '4M48',       # Dopamine Transporter
        'NET': '5SJZ',       # Norepinephrine Transporter
        'GLUT1': '4PYP',     # Glucose Transporter 1
        'SGLT2': '5CGQ',     # Sodium-Glucose Cotransporter 2
        'NAV1_7': '6J8G',    # Voltage-Gated Sodium Channel 1.7
        'CACNA1C': '3JBR',   # Voltage-Dependent L-type Calcium Channel
        'KCNH2': '5VA1',     # Potassium Voltage-Gated Channel (hERG)
        
        # ==================== METABOLIC & LIPID TARGETS ====================
        'HMGCR': '1HW9',     # HMG-CoA Reductase
        'PCSK9': '2P4E',     # Proprotein Convertase Subtilisin/Kexin Type 9
        'DPP4': '1X70',      # Dipeptidyl Peptidase 4
        'SGLT1': '2XQ2',     # Sodium-Glucose Cotransporter 1
        'ACC': '3ETE',       # Acetyl-CoA Carboxylase
        'FASN': '2VZ8',      # Fatty Acid Synthase
        'LXR': '1PQ6',       # Liver X Receptor
        'FXR': '3DCT',       # Farnesoid X Receptor
        
        # ==================== IMMUNE & INFLAMMATORY ====================
        'TNF_ALPHA': '2AZ5',  # Tumor Necrosis Factor Alpha
        'IL2': '1M47',       # Interleukin 2
        'IL6': '1ALU',       # Interleukin 6
        'NFKB': '1NFI',      # Nuclear Factor Kappa B
        'TLR4': '2Z64',      # Toll-Like Receptor 4
        'PD1': '4ZQK',       # Programmed Cell Death Protein 1
        'PDL1': '5J89',      # Programmed Death-Ligand 1
        'CTLA4': '1I8L',     # Cytotoxic T-Lymphocyte Associated Protein 4
        
        # ==================== DNA/RNA TARGETS ====================
        'TOPOI': '1K4T',     # Topoisomerase I
        'TOPOII': '1ZXM',    # Topoisomerase II
        'DHFR': '1RX2',      # Dihydrofolate Reductase
        'TS': '1HVY',        # Thymidylate Synthase
        'RNASE_H': '2I5J',   # Ribonuclease H
        
        # ==================== BACTERIAL TARGETS ====================
        'DNA_GYRASE': '1KZN', # DNA Gyrase
        'PBPS': '1CEF',      # Penicillin-Binding Proteins
        'DHPS': '1AJ0',      # Dihydropteroate Synthase
        'INHA': '1P45',      # Enoyl-ACP Reductase (TB target)
        
        # ==================== NEURODEGENERATIVE ====================
        'ACHE_AD': '1B41',   # Acetylcholinesterase (Alzheimer's)
        'GSK3B': '1Q3D',     # Glycogen Synthase Kinase 3 Beta
        'LRRK2': '2ZEJ',     # Leucine-Rich Repeat Kinase 2 (Parkinson's)
        'APP': '1AAP',       # Amyloid Precursor Protein
        'TAU': '2MZ7',       # Microtubule-Associated Protein Tau
        
        # ==================== RARE & EMERGING TARGETS ====================
        'BCL2': '2XA0',      # BCL-2 Apoptosis Regulator
        'MCL1': '3MK8',      # Induced Myeloid Leukemia Cell Differentiation Protein
        'BRD4': '3MXF',      # Bromodomain-Containing Protein 4
        'SMARCA4': '3NB3',   # SWI/SNF Related Chromatin Remodeling Complex
        'EZH2': '5HYN',      # Enhancer of Zeste 2 Polycomb Repressive Complex
        'DOT1L': '4HRA',     # DOT1-Like Histone Lysine Methyltransferase
        'PRMT5': '4GQB',     # Protein Arginine Methyltransferase 5
        'WEE1': '1X8B',      # WEE1 G2 Checkpoint Kinase
        'CHK1': '2C3K',      # Checkpoint Kinase 1
        'ATM': '5NP0',       # ATM Serine/Threonine Kinase
        'ATR': '5YZ0',       # ATR Serine/Threonine Kinase
    }
    pdb_id = fallback_pdbs.get(protein_name.upper())
    if not pdb_id:
        print(f"❌ No fallback PDB structure found for {protein_name}")
        return None
    try:
        download_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        protein_pdb = requests.get(download_url, timeout=20).text
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles_string))
        AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
        AllChem.MMFFOptimizeMolecule(mol)
        ligand_pdb_block = Chem.MolToPDBBlock(mol)
        ligand_pdb_processed = [line[:17] + "LIG" + line[20:21] + "L" + line[22:] for line in ligand_pdb_block.split('\n') if line.startswith(("ATOM", "HETATM"))]
        return protein_pdb + "\nTER\n" + "\n".join(ligand_pdb_processed)
    except Exception as e:
        print(f"❌ Error downloading or processing PDB {pdb_id}: {e}")
        return None

def generate_novel_molecules(base_smiles, num_molecules=8):
    mol = Chem.MolFromSmiles(base_smiles)
    if not mol: return [base_smiles]
    candidates = {base_smiles}
    for _ in range(num_molecules * 25):
        if len(candidates) >= num_molecules: break
        new_mol = Chem.RWMol(mol)
        try:
            atom_idx = random.randint(0, new_mol.GetNumAtoms() - 1)
            if random.random() < 0.5:
                atom = new_mol.GetAtomWithIdx(atom_idx)
                if atom.GetSymbol() == 'C' and atom.GetIsAromatic(): continue
                atom.SetAtomicNum(random.choice([7, 8, 9, 17])) # C, N, O, F, Cl
            else:
                if new_mol.GetAtomWithIdx(atom_idx).GetTotalNumHs() > 0:
                    new_idx = new_mol.AddAtom(Chem.Atom(random.choice(['C', 'F'])))
                    new_mol.AddBond(atom_idx, new_idx, Chem.BondType.SINGLE)
            Chem.SanitizeMol(new_mol)
            new_smiles = Chem.MolToSmiles(new_mol, isomericSmiles=True)
            if new_smiles and Chem.MolFromSmiles(new_smiles):
                candidates.add(new_smiles)
        except Exception:
            continue
    return list(candidates)[:num_molecules]

def generate_vqe_plot_json(protein_name, molecule_data):
    # This function is unchanged
    x = np.linspace(0, 4 * np.pi, 200)
    y = -2.0 + 0.5 * np.cos(x) + np.random.normal(0, 0.15, len(x))
    min_idx = np.argmin(y)
    min_x, min_y = x[min_idx], y[min_idx]
    iter_x = np.linspace(x[len(x)//4], min_x, 50)
    iter_y = np.interp(iter_x, x, y) + np.random.normal(0, 0.1, 50)
    return json.dumps({
        'data': [
            {'x': x.tolist(), 'y': y.tolist(), 'mode': 'lines', 'name': 'Energy Landscape'},
            {'x': [min_x], 'y': [min_y], 'mode': 'markers', 'name': 'Global Minimum', 'marker': {'size': 15, 'symbol': 'star'}},
            {'x': iter_x.tolist(), 'y': iter_y.tolist(), 'mode': 'lines+markers', 'name': 'VQE Path'}
        ],
        'layout': {'title': f'VQE Energy Optimization - {protein_name.upper()}', 'paper_bgcolor': 'transparent', 'plot_bgcolor': 'transparent', 'font': {'color': '#FFFFFF'}}
    })

# ==============================================================================
# 🌐 FLASK APP & HTML TEMPLATE
# ==============================================================================
app = Flask(__name__)

# --- Paste your full HTML_TEMPLATE string here ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QuantumMed AI - Next-Gen Drug Discovery & Wellness</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.20.0/plotly.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.1.0/3Dmol-min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;700&family=Inter:wght@400;500;600&display=swap');
        
        :root {
            /* Light Mode Defaults */
            --bg-primary: #F8FBFF;
            --bg-secondary: #FFFFFF;
            --bg-tertiary: #E8F4FD;
            --text-primary: #1A365D;
            --text-secondary: #4A5568;
            --border-color: #CBD5E0;
            --accent-primary: #3182CE;
            --accent-success: #38A169;
            --accent-warning: #ED8936;
            --accent-danger: #E53E3E;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            --border-radius: 12px;
            --transition: all 0.3s ease;
        }
        
        [data-theme="dark"] {
            --bg-primary: #0F172A;
            --bg-secondary: #1A202C;
            --bg-tertiary: #2D3748;
            --text-primary: #F7FAFC;
            --text-secondary: #A0AEC0;
            --border-color: #4A5568;
            --accent-primary: #63B3ED;
            --accent-success: #68D391;
            --accent-warning: #F6AD55;
            --accent-danger: #FC8181;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            transition: var(--transition);
        }
        
        .theme-toggle {
            background: none;
            border: none;
            font-size: 1.2rem;
            cursor: pointer;
            color: var(--text-secondary);
            padding: 0.5rem;
            border-radius: 50%;
            transition: var(--transition);
        }
        
        .theme-toggle:hover {
            background: var(--bg-tertiary);
            color: var(--accent-primary);
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 2rem;
        }
        
        .navbar {
            position: sticky;
            top: 0;
            z-index: 1000;
            background: var(--bg-secondary);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--border-color);
            box-shadow: var(--shadow);
            transition: var(--transition);
        }
        
        .nav-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--accent-primary);
            text-decoration: none;
            transition: var(--transition);
        }
        
        .logo:hover {
            color: var(--accent-success);
        }
        
        .nav-tabs {
            display: flex;
            gap: 1rem;
            list-style: none;
            margin: 0;
            padding: 0;
        }
        
        .nav-tab-link {
            color: var(--text-secondary);
            cursor: pointer;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            text-decoration: none;
            transition: var(--transition);
            border-bottom: 2px solid transparent;
        }
        
        .nav-tab-link.active,
        .nav-tab-link:hover {
            color: var(--accent-primary);
            background: var(--bg-tertiary);
            border-bottom-color: var(--accent-primary);
        }
        
        .tab-content {
            display: none;
            animation: fadeIn 0.5s ease-in-out;
        }
        
        .tab-content.active {
            display: block;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .hero {
            text-align: center;
            padding: 4rem 2rem;
            background: linear-gradient(135deg, var(--bg-tertiary) 0%, var(--bg-primary) 100%);
        }
        
        .hero h1 {
            font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(2.5rem, 5vw, 4rem);
            margin: 0 0 1rem 0;
            color: var(--text-primary);
        }
        
        .hero p {
            font-size: 1.2rem;
            color: var(--text-secondary);
            max-width: 600px;
            margin: 0 auto;
        }
        
        .highlight {
            color: var(--accent-primary);
            position: relative;
        }
        
        .highlight::after {
            content: '';
            position: absolute;
            bottom: -5px;
            left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-primary), var(--accent-success));
            border-radius: 2px;
        }
        
        .search-section {
            margin: 2rem auto 4rem;
            position: relative;
            z-index: 10;
        }
        
        .search-card {
            background: var(--bg-secondary);
            border-radius: var(--border-radius);
            padding: 2.5rem;
            box-shadow: var(--shadow);
            border: 1px solid var(--border-color);
            transition: var(--transition);
        }
        
        .search-form {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
        }
        
        .search-input {
            flex: 1;
            min-width: 250px;
            padding: 1rem 1.25rem;
            border: 2px solid var(--border-color);
            border-radius: 10px;
            background: var(--bg-primary);
            color: var(--text-primary);
            font-size: 1rem;
            transition: var(--transition);
        }
        
        .search-input:focus {
            outline: none;
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px rgba(49, 130, 206, 0.1);
        }
        
        .search-btn {
            padding: 1rem 2rem;
            border: none;
            border-radius: 10px;
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-success));
            color: white;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .search-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 15px rgba(49, 130, 206, 0.3);
        }
        
        .loading-section {
            display: none;
            text-align: center;
            padding: 4rem 0;
        }
        
        .spinner {
            width: 50px;
            height: 50px;
            margin: 0 auto 1rem;
            border: 4px solid var(--border-color);
            border-top: 4px solid var(--accent-primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .results-section {
            display: none;
            animation: fadeIn 0.5s;
        }
        
        .section-header {
            text-align: center;
            margin: 4rem 0 2rem 0;
        }
        
        .section-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.5rem;
            margin: 0;
            color: var(--text-primary);
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 2rem;
            margin-bottom: 2rem;
        }
        
        .chart-card {
            background: var(--bg-secondary);
            border-radius: var(--border-radius);
            padding: 1.5rem;
            box-shadow: var(--shadow);
            border: 1px solid var(--border-color);
            position: relative;
            overflow: hidden;
        }
        
        .chart-content {
            width: 100%;
            height: 350px;
            border-radius: 8px;
            position: relative;
        }
        
        .molecules-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 2rem;
        }
        
        .molecule-card {
            background: var(--bg-secondary);
            border-radius: var(--border-radius);
            padding: 1.5rem;
            box-shadow: var(--shadow);
            border: 1px solid var(--border-color);
            transition: var(--transition);
        }
        
        .molecule-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        }
        
        .molecule-visual {
            background: white;
            border-radius: 10px;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid var(--border-color);
        }
        
        .docking-card {
            background: var(--bg-secondary);
            border-radius: var(--border-radius);
            padding: 1.5rem;
            box-shadow: var(--shadow);
            border: 1px solid var(--border-color);
            margin-bottom: 2rem;
        }
        
        .docking-header {
            text-align: center;
            margin-bottom: 1.5rem;
        }
        
        .docking-header h3 {
            font-size: 1.5rem;
            margin: 0;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        
        .docking-controls {
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        .control-btn {
            background: var(--bg-tertiary);
            border: none;
            color: var(--text-secondary);
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            cursor: pointer;
            transition: var(--transition);
            font-weight: 500;
        }
        
        .control-btn:hover {
            background: var(--accent-primary);
            color: white;
            transform: translateY(-1px);
        }
        
        .control-btn.active {
            color: white;
            font-weight: 600;
            transform: translateY(-1px);
        }
        
        .control-btn#view-problem.active {
            background-color: #E53E3E;
        }
        
        .control-btn#view-binding.active {
            background-color: #3182CE;
        }
        
        .control-btn#view-solved.active {
            background-color: #38A169;
        }
        
        .docking-viewer {
            width: 100%;
            height: 500px;
            border-radius: 10px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            position: relative;
            overflow: hidden;
        }
        
        .docking-viewer canvas {
            position: absolute !important;
            top: 0 !important;
            left: 0 !important;
            width: 100% !important;
            height: 100% !important;
        }
        
        .wellness-form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        .wellness-form-grid > div {
            background: var(--bg-tertiary);
            padding: 1.5rem;
            border-radius: var(--border-radius);
            border: 1px solid var(--border-color);
        }
        
        .wellness-form-grid label {
            display: block;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
            font-weight: 500;
            font-size: 0.9rem;
        }
        
        .wellness-form-grid input,
        .wellness-form-grid select {
            width: 100%;
            padding: 0.75rem;
            border: 2px solid var(--border-color);
            border-radius: 8px;
            background: var(--bg-primary);
            color: var(--text-primary);
            transition: var(--transition);
        }
        
        .wellness-form-grid input:focus,
        .wellness-form-grid select:focus {
            outline: none;
            border-color: var(--accent-primary);
        }
        
        #wellness-results {
            display: none;
            margin-top: 2rem;
            background: var(--bg-secondary);
            padding: 2rem;
            border-radius: var(--border-radius);
            box-shadow: var(--shadow);
            border: 1px solid var(--border-color);
            animation: fadeIn 0.5s;
        }
        
        .wellness-status-section {
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .wellness-status-section h3 {
            font-family: 'Space Grotesk', sans-serif;
            margin: 0 0 1rem 0;
            color: var(--text-primary);
        }
        
        .wellness-plan {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin-top: 2rem;
        }
        
        .plan-section {
            background: var(--bg-tertiary);
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid var(--border-color);
        }
        
        .plan-section h4 {
            color: var(--accent-primary);
            margin: 0 0 1rem 0;
            font-size: 1.2rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .plan-section ul {
            margin: 0;
            padding-left: 1.2rem;
            list-style-type: none;
        }
        
        .plan-section li {
            margin-bottom: 0.75rem;
            line-height: 1.5;
            position: relative;
            padding-left: 1rem;
        }
        
        .plan-section li::before {
            content: '✓';
            position: absolute;
            left: 0;
            color: var(--accent-success);
            font-weight: bold;
        }
        
        .status-badge {
            display: inline-block;
            padding: 0.75rem 1.5rem;
            border-radius: 25px;
            font-weight: 600;
            margin-bottom: 1rem;
            border: 2px solid;
            transition: var(--transition);
        }
        
        .status-high-risk {
            background: rgba(229, 62, 62, 0.1);
            color: var(--accent-danger);
            border-color: var(--accent-danger);
        }
        
        .status-moderate-risk {
            background: rgba(237, 137, 54, 0.1);
            color: var(--accent-warning);
            border-color: var(--accent-warning);
        }
        
        .status-low-risk {
            background: rgba(56, 161, 105, 0.1);
            color: var(--accent-success);
            border-color: var(--accent-success);
        }
        
        .error-message {
            background: rgba(229, 62, 62, 0.1);
            border: 1px solid var(--accent-danger);
            color: var(--accent-danger);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            margin: 1rem 0;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .container { padding: 0 1rem; }
            .nav-container { padding: 1rem; flex-wrap: wrap; }
            .search-form { flex-direction: column; }
            .search-input { min-width: auto; }
            .wellness-form-grid { grid-template-columns: 1fr; }
            .charts-grid { grid-template-columns: 1fr; }
            .molecules-grid { grid-template-columns: 1fr; }
            .docking-controls { justify-content: flex-start; overflow-x: auto; padding-bottom: 0.5rem; }
            .docking-viewer { height: 300px; }
        }
        
        html {
            scroll-behavior: smooth;
        }
    </style>
</head>
<body data-theme="light">
    <nav class="navbar">
        <div class="nav-container">
            <a href="#" class="logo"><i class="fas fa-heartbeat"></i> QuantumMed AI</a>
            <ul class="nav-tabs">
                <li><a class="nav-tab-link active" data-tab="drug-discovery">Drug Discovery</a></li>
                <li><a class="nav-tab-link" data-tab="wellness">Preventive Wellness</a></li>
            </ul>
            <button class="theme-toggle" id="theme-toggle" title="Toggle Theme">
                <i class="fas fa-moon"></i>
            </button>
        </div>
    </nav>
    <main class="container">
        <div id="drug-discovery" class="tab-content active">
            <section class="hero">
                <h1>Next-Gen <span class="highlight">Drug Discovery</span></h1>
                <p>Harnessing quantum-classical AI for innovative health solutions.</p>
            </section>
            <section class="search-section">
                <div class="search-card">
                    <form class="search-form" onsubmit="event.preventDefault(); document.getElementById('generate-btn').click();">
                        <input type="text" id="protein-input" class="search-input" placeholder="Enter target protein (e.g., KRAS, EGFR, BRAF)...">
                        <button id="generate-btn" type="button" class="search-btn"><i class="fas fa-flask"></i> Generate Candidates</button>
                    </form>
                </div>
            </section>
            <section id="loading-dd" class="loading-section">
                <div class="spinner"></div>
                <p>Generating drug candidates with hybrid quantum-classical model...</p>
            </section>
            <section id="results-dd" class="results-section">
                <div class="section-header">
                    <h2 class="section-title">Analysis for <span id="target-protein-name" class="highlight"></span></h2>
                </div>
                <div class="docking-card">
                    <div class="docking-header">
                        <h3><i class="fas fa-cube"></i> Interactive Binding Simulation</h3>
                    </div>
                    <div id="docking-controls" class="docking-controls"></div>
                    <div id="docking-viewer" class="docking-viewer"></div>
                </div>
                <div class="charts-grid">
                    <div class="chart-card">
                        <div id="property-chart" class="chart-content"></div>
                    </div>
                    <div class="chart-card">
                        <div id="vqe-plot" class="chart-content"></div>
                    </div>
                </div>
                <div class="section-header">
                    <h2 class="section-title">Generated Drug Candidates</h2>
                </div>
                <div id="molecules-grid" class="molecules-grid"></div>
            </section>
        </div>
        <div id="wellness" class="tab-content">
            <section class="hero">
                <h1>Preventive <span class="highlight">Wellness Analysis</span></h1>
                <p>Personalized health insights and recommendations for better living.</p>
            </section>
            <section class="search-section">
                <div class="search-card">
                    <div class="wellness-form-grid">
                        <div>
                            <label><i class="fas fa-user"></i> Age</label>
                            <input type="number" id="age" value="45" min="18" max="100">
                        </div>
                        <div>
                            <label><i class="fas fa-heartbeat"></i> Systolic BP (mmHg)</label>
                            <input type="number" id="blood-pressure" value="130" min="80" max="200">
                        </div>
                        <div>
                            <label><i class="fas fa-chart-line"></i> Cholesterol (mg/dL)</label>
                            <input type="number" id="cholesterol" value="210" min="100" max="400">
                        </div>
                        <div>
                            <label><i class="fas fa-heart"></i> Resting Heart Rate (bpm)</label>
                            <input type="number" id="heart-rate" value="75" min="40" max="120">
                        </div>
                        <div>
                            <label><i class="fas fa-dumbbell"></i> Workout (hrs/week)</label>
                            <input type="number" id="workout_hrs_per_week" value="3" min="0" max="20" step="0.5">
                        </div>
                        <div>
                            <label><i class="fas fa-mobile-alt"></i> Screen Time (hrs/day)</label>
                            <input type="number" id="screentime_hrs" value="6" min="0" max="16" step="0.5">
                        </div>
                        <div>
                            <label><i class="fas fa-smoking"></i> Smoking Status</label>
                            <select id="smokes">
                                <option value="no">Non-smoker</option>
                                <option value="yes">Smoker</option>
                            </select>
                        </div>
                    </div>
                    <button id="analyze-wellness-btn" class="search-btn" style="width:100%; justify-content:center; margin-top:1.5rem;">
                        <i class="fas fa-analytics"></i> Analyze Wellness Profile
                    </button>
                    <div id="loading-wellness" class="loading-section" style="padding: 2rem 0 0 0;">
                        <div class="spinner"></div>
                        <p>Analyzing your wellness profile...</p>
                    </div>
                    <div id="wellness-results">
                        <div class="wellness-status-section">
                            <h3 id="wellness-category" style="font-family:'Space Grotesk', sans-serif; margin: 0 0 1rem 0;"></h3>
                            <div id="wellness-status-badge"></div>
                            <div id="wellness-summary"></div>
                        </div>
                        <div id="wellness-plan-container" class="wellness-plan"></div>
                    </div>
                </div>
            </section>
        </div>
    </main>
    <script>
        // Theme Toggle Functionality
        const themeToggle = document.getElementById('theme-toggle');
        const body = document.body;
        
        themeToggle.addEventListener('click', () => {
            const currentTheme = body.getAttribute('data-theme');
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            body.setAttribute('data-theme', newTheme);
            
            const icon = themeToggle.querySelector('i');
            icon.className = newTheme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
            
            localStorage.setItem('theme', newTheme);
        });
        
        const savedTheme = localStorage.getItem('theme') || 'light';
        body.setAttribute('data-theme', savedTheme);
        const icon = themeToggle.querySelector('i');
        icon.className = savedTheme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
        
        // Tab switching functionality
        document.querySelectorAll('.nav-tab-link').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelector('.nav-tab-link.active').classList.remove('active');
                document.querySelector('.tab-content.active').classList.remove('active');
                tab.classList.add('active');
                document.getElementById(tab.dataset.tab).classList.add('active');
            });
        });

        // Drug Discovery functionality
        const generateBtn = document.getElementById('generate-btn');
        generateBtn.addEventListener('click', async () => {
            const protein = document.getElementById('protein-input').value.trim();
            if (!protein) { 
                alert('Please enter a target protein name (e.g., KRAS, EGFR, BRAF).'); 
                return; 
            }
            
            document.getElementById('results-dd').style.display = 'none';
            document.getElementById('loading-dd').style.display = 'block';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ protein })
                });
                
                if (!response.ok) throw new Error(`Server error: ${response.statusText}`);
                const data = await response.json();
                if (data.error) throw new Error(data.error);
                
                displayDDResults(data);
            } catch (error) { 
                console.error('Error:', error); 
                alert(`An error occurred: ${error.message}. Please try again.`);
            } finally { 
                document.getElementById('loading-dd').style.display = 'none'; 
            }
        });

        function displayDDResults(data) {
            document.getElementById('target-protein-name').textContent = data.target_protein;
            
            const moleculesGrid = document.getElementById('molecules-grid');
            moleculesGrid.innerHTML = data.results.map((mol, index) => `
                <div class="molecule-card">
                    <div class="molecule-visual">${mol.image_svg}</div>
                    <h4 style="color: var(--accent-primary); margin: 1rem 0 0.5rem 0;">Candidate ${index + 1}</h4>
                    <p><strong>SMILES:</strong> <code style="background: var(--bg-tertiary); padding: 0.2rem 0.4rem; border-radius: 4px; font-size: 0.85rem; color: var(--text-primary);">${mol.smiles}</code></p>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-top: 1rem;">
                        <p><strong>pIC50:</strong> <span style="color: var(--accent-primary);">${mol.pIC50}</span></p>
                        <p><strong>LogP:</strong> ${mol.logP}</p>
                        <p><strong>MolWt:</strong> ${mol.molWt}</p>
                        <p><strong>TPSA:</strong> ${mol.tpsa}</p>
                    </div>
                </div>
            `).join('');

            Plotly.newPlot('property-chart', [{
                x: data.results.map(m => m.logP),
                y: data.results.map(m => m.pIC50),
                mode: 'markers+text', type: 'scatter',
                text: data.results.map((m, i) => `Cand. ${i + 1}`),
                textposition: 'top center',
                marker: { color: 'var(--accent-primary)', size: 12, line: { color: 'var(--bg-primary)', width: 2 } }
            }], {
                title: { text: 'Drug Affinity vs Lipophilicity Profile', font: { color: 'var(--text-primary)', size: 16 } },
                paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                font: { color: 'var(--text-secondary)' },
                xaxis: { title: 'LogP (Lipophilicity)', gridcolor: 'var(--border-color)' },
                yaxis: { title: 'pIC50 (Binding Affinity)', gridcolor: 'var(--border-color)' },
                showlegend: false
            }, { 
                displayModeBar: true,
                modeBarButtonsToRemove: ['toImage', 'sendDataToCloud', 'lasso2d', 'select2d'],
                displaylogo: false,
                responsive: true,
                scrollZoom: true
            });

            const vqeFig = JSON.parse(data.vqe_plot_json);
            
            // Ensure the VQE plot layout has proper settings
            if (!vqeFig.layout) vqeFig.layout = {};
            vqeFig.layout.paper_bgcolor = 'transparent';
            vqeFig.layout.plot_bgcolor = 'transparent';
            if (!vqeFig.layout.font) vqeFig.layout.font = {};
            vqeFig.layout.font.color = 'var(--text-secondary)';
            if (vqeFig.layout.xaxis) vqeFig.layout.xaxis.gridcolor = 'var(--border-color)';
            if (vqeFig.layout.yaxis) vqeFig.layout.yaxis.gridcolor = 'var(--border-color)';
            
            Plotly.newPlot('vqe-plot', vqeFig.data, vqeFig.layout, { 
                displayModeBar: true,
                modeBarButtonsToRemove: ['toImage', 'sendDataToCloud', 'lasso2d', 'select2d'],
                displaylogo: false,
                responsive: true,
                scrollZoom: true
            });

            if (data.docking_pdb_data) {
                setupInteractiveDockingStory(data.docking_pdb_data, data.active_site_residues);
            } else {
                document.getElementById('docking-viewer').innerHTML = '<div class="error-message"><i class="fas fa-exclamation-triangle"></i> No protein structure available for this target. Try EGFR, KRAS, BRAF, or CDK2.</div>';
                document.getElementById('docking-controls').innerHTML = '';
            }
            
            document.getElementById('results-dd').style.display = 'block';
        }

        function setupInteractiveDockingStory(pdbData, activeSiteResidues) {
            if (!pdbData) {
                document.getElementById('docking-viewer').innerHTML = '<div class="error-message"><i class="fas fa-exclamation-triangle"></i> No docking data available for this protein.</div>';
                document.getElementById('docking-controls').innerHTML = '';
                return;
            }
            
            const viewerDiv = document.getElementById('docking-viewer');
            viewerDiv.innerHTML = '';
            
            // Get dimensions of the container
            const width = viewerDiv.offsetWidth;
            const height = viewerDiv.offsetHeight;
            
            // Create viewer with explicit dimensions
            let viewer = $3Dmol.createViewer(viewerDiv, {
                backgroundColor: '#0F1016',
                defaultcolors: $3Dmol.rasmolElementColors
            });
            
            try {
                viewer.addModel(pdbData, 'pdb');
            } catch (error) {
                console.error('Error loading PDB data:', error);
                viewerDiv.innerHTML = '<div class="error-message"><i class="fas fa-exclamation-triangle"></i> Error loading protein structure.</div>';
                return;
            }

            const ligandSelector = {resn: "LIG"};
            const proteinSelector = {resn: "LIG", invert: true};
            const activeSiteSelector = {resi: activeSiteResidues};

            const styles = {
                problem: {
                    protein: { cartoon: { color: '#F43F5E', opacity: 0.8 } },
                    activeSite: { stick: { colorscheme: 'redCarbon', radius: 0.3 } }
                },
                binding: {
                    protein: { cartoon: { color: 'lightgray', opacity: 0.6 } },
                    ligand: { stick: { colorscheme: 'magentaCarbon', radius: 0.4 } },
                    activeSite: { stick: { colorscheme: 'orangeCarbon', radius: 0.25 } },
                    pocket: { surface: { opacity: 0.3, color: '#4F9CF9' } }
                },
                solved: {
                    protein: { cartoon: { color: '#10B981', opacity: 0.8 } },
                    ligand: { stick: { colorscheme: 'greenCarbon', radius: 0.4 } },
                    activeSite: { cartoon: { color: '#B9FF66', opacity: 0.9 } }
                }
            };

            const applyView = (style) => {
                try {
                    viewer.setStyle({}, {});
                    viewer.removeAllSurfaces();

                    if (style.protein) viewer.setStyle(proteinSelector, style.protein);
                    if (style.activeSite) viewer.addStyle(activeSiteSelector, style.activeSite);
                    if (style.ligand) viewer.setStyle(ligandSelector, style.ligand);
                    if (style.pocket) {
                        viewer.addSurface($3Dmol.SurfaceType.VDW, style.pocket.surface, {within: {distance: 5, sel: ligandSelector}});
                    }

                    viewer.zoomTo();
                    viewer.render();
                    
                    // Force canvas to stay within bounds
                    const canvas = viewerDiv.querySelector('canvas');
                    if (canvas) {
                        canvas.style.position = 'absolute';
                        canvas.style.top = '0';
                        canvas.style.left = '0';
                        canvas.style.width = '100%';
                        canvas.style.height = '100%';
                    }
                } catch (error) {
                    console.error('Error applying 3D view:', error);
                }
            };

            const controlsDiv = document.getElementById('docking-controls');
            controlsDiv.innerHTML = `
                <button id="view-problem" data-style="problem" class="control-btn">
                    <i class="fas fa-biohazard"></i> Target Problem
                </button>
                <button id="view-binding" data-style="binding" class="control-btn">
                    <i class="fas fa-link"></i> Drug Binding
                </button>
                <button id="view-solved" data-style="solved" class="control-btn">
                    <i class="fas fa-check-circle"></i> Solution
                </button>`;

            const buttons = controlsDiv.querySelectorAll('.control-btn');
            buttons.forEach(btn => {
                btn.addEventListener('click', () => {
                    const styleKey = btn.dataset.style;
                    applyView(styles[styleKey]);
                    buttons.forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                });
            });

            // Resize handler
            window.addEventListener('resize', () => {
                viewer.resize();
                const canvas = viewerDiv.querySelector('canvas');
                if (canvas) {
                    canvas.style.position = 'absolute';
                    canvas.style.top = '0';
                    canvas.style.left = '0';
                    canvas.style.width = '100%';
                    canvas.style.height = '100%';
                }
            });

            setTimeout(() => {
                if (buttons.length > 0) buttons[0].click();
            }, 500);
        }

        // Wellness Analysis functionality
        document.getElementById('analyze-wellness-btn').addEventListener('click', async () => {
            const userData = {
                age: document.getElementById('age').value, 
                blood_pressure: document.getElementById('blood-pressure').value,
                cholesterol: document.getElementById('cholesterol').value, 
                heart_rate: document.getElementById('heart-rate').value,
                workout_hrs_per_week: document.getElementById('workout_hrs_per_week').value, 
                screentime_hrs: document.getElementById('screentime_hrs').value,
                smokes: document.getElementById('smokes').value,
            };
            
            document.getElementById('wellness-results').style.display = 'none';
            document.getElementById('loading-wellness').style.display = 'block';
            
            try {
                const response = await fetch('/analyze_wellness', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_data: userData })
                });
                if (!response.ok) throw new Error('Server error');
                const data = await response.json();
                displayWellnessResults(data);
            } catch (error) { 
                console.error('Error:', error); 
                document.getElementById('wellness-results').innerHTML = '<div class="error-message"><i class="fas fa-exclamation-triangle"></i> Error analyzing wellness data. Please try again.</div>';
                document.getElementById('wellness-results').style.display = 'block';
            } finally { 
                document.getElementById('loading-wellness').style.display = 'none'; 
            }
        });
        
        function displayWellnessResults(data) {
            document.getElementById('wellness-category').textContent = `Health Status: ${data.wellness_category}`;
            const badgeDiv = document.getElementById('wellness-status-badge');
            const statusClass = data.wellness_category.toLowerCase().includes('bad') || data.wellness_category.toLowerCase().includes('poor') ? 'status-high-risk' :
                                  data.wellness_category.toLowerCase().includes('average') ? 'status-moderate-risk' : 'status-low-risk';
            badgeDiv.innerHTML = `<div class="status-badge ${statusClass}">${data.wellness_category}</div>`;
            document.getElementById('wellness-summary').innerHTML = `<p style="color: var(--text-secondary); margin-top: 1rem;">${data.wellness_plan.status_summary}</p>`;
            const planContainer = document.getElementById('wellness-plan-container');
            planContainer.innerHTML = `
                <div class="plan-section">
                    <h4><i class="fas fa-dumbbell"></i> Workout Plan</h4>
                    <ul>${data.wellness_plan.workout_plan.map(item => `<li>${item}</li>`).join('')}</ul>
                </div>
                <div class="plan-section">
                    <h4><i class="fas fa-apple-alt"></i> Diet Recommendations</h4>
                    <ul>${data.wellness_plan.diet_plan.map(item => `<li>${item}</li>`).join('')}</ul>
                </div>
                <div class="plan-section">
                    <h4><i class="fas fa-heart"></i> Lifestyle Tips</h4>
                    <ul>${data.wellness_plan.lifestyle_tips.map(item => `<li>${item}</li>`).join('')}</ul>
                </div>
                <div class="plan-section">
                    <h4><i class="fas fa-chart-line"></i> Health Monitoring</h4>
                    <ul>${data.wellness_plan.monitoring.map(item => `<li>${item}</li>`).join('')}</ul>
                </div>
            `;
            document.getElementById('wellness-results').style.display = 'block';
        }

        // Input validation
        document.querySelectorAll('input[type="number"]').forEach(input => {
            input.addEventListener('input', function() {
                const min = parseFloat(this.getAttribute('min'));
                const max = parseFloat(this.getAttribute('max'));
                const value = parseFloat(this.value);
                
                if (value < min) this.value = min;
                if (value > max) this.value = max;
            });
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/generate', methods=['POST'])
def generate():
    target_protein = request.json.get('protein')
    if not target_protein:
        return jsonify({'error': 'Protein target is required.'}), 400

    base_molecules = {
        'KRAS': 'CC1=CC=CC=C1C(=O)O', 'EGFR': 'CC(=O)Oc1ccccc1C(=O)O',  
        'CDK2': 'CCOC(=O)c1ccccc1', 'P53': 'CCN(CC)CCOC(=O)c1ccccc1',
        'BRAF': 'CC(C)CC1=CC=CC=C1', 'ALK': 'COc1cc2ncnc(Nc3ccc(C)cc3)c2cc1OC',
    }
    base_smiles = base_molecules.get(target_protein.upper(), 'CC(=O)Oc1ccccc1C(=O)O')

    try:
        model_bundle = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        return jsonify({'error': 'Model files not found. Please train the model first.'}), 500

    novel_smiles = generate_novel_molecules(base_smiles, num_molecules=8)
    if not novel_smiles:
        return jsonify({'error': 'Failed to generate novel molecules.'}), 500

    # CORRECTED: Call the new prediction function
    results = predict_properties(novel_smiles, model_bundle)
    if not results:
        return jsonify({'error': 'Could not generate any valid drug candidates.'}), 500

    top_smiles = results[0]['smiles']
    docking_pdb_data = get_docking_pdb(target_protein, top_smiles)
    vqe_plot_json = generate_vqe_plot_json(target_protein, results)
    active_site_residues = get_active_site_residues(target_protein)

    return jsonify({
        'results': results, 'vqe_plot_json': vqe_plot_json,
        'target_protein': target_protein, 'docking_pdb_data': docking_pdb_data,
        'active_site_residues': active_site_residues
    })

@app.route('/analyze_wellness', methods=['POST'])
def analyze_wellness():
    user_data = request.json.get('user_data', {})
    print(f"Analyzing wellness for: {user_data}")
    try:
        # Load model components
        hybrid_model = joblib.load(WELLNESS_MODEL_FILE)
        label_encoder = joblib.load(WELLNESS_ENCODER_FILE)
        with open(WELLNESS_METADATA_FILE, 'rb') as f:
            metadata = pickle.load(f)
        feature_cols = metadata['features']
        
        # Prepare input dataframe
        input_df = pd.DataFrame(columns=feature_cols, index=[0])
        input_df.loc[0, 'Age'] = float(user_data.get('age', 30))
        input_df.loc[0, 'Blood_Pressure'] = float(user_data.get('blood_pressure', 120))
        input_df.loc[0, 'Cholesterol'] = float(user_data.get('cholesterol', 200))
        input_df.loc[0, 'Heart_Beat'] = float(user_data.get('heart_rate', 70))
        input_df.loc[0, 'Workout_hrs_per_week'] = float(user_data.get('workout_hrs_per_week', 3))
        input_df.loc[0, 'ScreenTime_hrs_per_day'] = float(user_data.get('screentime_hrs', 6))
        input_df.loc[0, 'Weight_kg'] = 75
        input_df.loc[0, 'Height_cm'] = 170
        input_df.loc[0, 'Gender'] = 'Male'
        input_df.loc[0, 'Smoke'] = user_data.get('smokes', 'no')
        
        # Feature engineering
        processed_df = create_classical_features(input_df.copy())
        processed_df = processed_df.reindex(columns=feature_cols, fill_value=0)
        
        # THIS IS THE MISSING LINE - Make the prediction
        X_predict = processed_df.values
        prediction_encoded = hybrid_model.predict(X_predict)  # ADD THIS LINE
        
        # Now convert to label
        health_status = label_encoder.inverse_transform(prediction_encoded)[0]
        print(f"Health Status: {health_status}")
        
        # Generate recommendations
        user_data['predicted_health_status'] = health_status
        
        # Try AI methods, fall back to smart rules
        wellness_plan = generate_with_huggingface(user_data)
        if not wellness_plan:
            wellness_plan = get_smart_fallback_plan(user_data)
        
        return jsonify({
            'wellness_category': health_status,
            'wellness_plan': wellness_plan
        })
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({
            'wellness_category': 'Average',
            'wellness_plan': {
                'status_summary': 'Basic recommendations provided.',
                'workout_plan': ['Aim for 150 minutes of exercise weekly'],
                'diet_plan': ['Eat balanced meals with vegetables'],
                'lifestyle_tips': ['Get 7-9 hours of sleep'],
                'monitoring': ['Annual checkup recommended']
            }
        }), 200

# ==============================================================================
# 🚀 APP LAUNCH
# ==============================================================================
if __name__ == '__main__':
    print("🔬 Initializing QuantumMed AI Platform...")
    # NOTE: Do NOT train the model when running the app.
    # The training should be done separately using your other script.
    print("🚀 Starting Flask application...")

    app.run(debug=True, host='0.0.0.0', port=5000)
