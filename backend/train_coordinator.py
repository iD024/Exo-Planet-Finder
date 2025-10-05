import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import os

# --- Configuration ---
KOI_FILE_PATH = 'KOI.csv'
TOI_FILE_PATH = 'TOI.csv'
MODEL_OUTPUT_DIR = 'models'
MODEL_OUTPUT_PATH = os.path.join(MODEL_OUTPUT_DIR, 'coordinator_model.joblib')

def load_and_prepare_koi_data(filepath: str) -> pd.DataFrame:
    print(f"Loading KOI data from {filepath}...")
    try:
        koi_df = pd.read_csv(filepath, comment='#')
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return pd.DataFrame()

    features = {
        'koi_disposition': 'target', 'koi_period': 'period', 'koi_duration': 'duration',
        'koi_depth': 'depth', 'koi_steff': 'stellar_teff', 'koi_slogg': 'stellar_logg',
        'koi_impact': 'impact'
    }
    koi_df = koi_df[features.keys()].rename(columns=features)
    koi_df['target'] = koi_df['target'].apply(lambda x: 1 if x in ['CONFIRMED', 'CANDIDATE'] else 0)
    koi_df['impact'] = koi_df['impact'].fillna(0.5)
    
    koi_df = koi_df.dropna()
    print(f"Loaded and prepared {len(koi_df)} clean rows from KOI data.")
    return koi_df

def load_and_prepare_toi_data(filepath: str) -> pd.DataFrame:
    print(f"Loading TOI data from {filepath}...")
    try:
        toi_df = pd.read_csv(filepath, comment='#')
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return pd.DataFrame()

    # --- FIXED: Handle potentially missing 'pl_imppar' column ---
    base_features = {
        'tfopwg_disp': 'target', 'pl_orbper': 'period', 'pl_trandurh': 'duration',
        'pl_trandep': 'depth', 'st_teff': 'stellar_teff', 'st_logg': 'stellar_logg',
    }
    
    # Check if the impact parameter column exists in the CSV
    if 'pl_imppar' in toi_df.columns:
        print("Found 'pl_imppar' (impact parameter) in TOI data.")
        base_features['pl_imppar'] = 'impact'
        toi_df = toi_df[base_features.keys()].rename(columns=base_features)
        toi_df['impact'] = toi_df['impact'].fillna(0.5)
    else:
        print("Warning: 'pl_imppar' (impact parameter) not found in TOI data. Using a placeholder value.")
        toi_df = toi_df[base_features.keys()].rename(columns=base_features)
        # Create a placeholder column if it doesn't exist
        toi_df['impact'] = 0.5

    toi_df['target'] = toi_df['target'].apply(lambda x: 1 if x == 'PC' else 0)
    toi_df = toi_df.dropna()
    print(f"Loaded and prepared {len(toi_df)} clean rows from TOI data.")
    return toi_df

def train_and_evaluate_model(df: pd.DataFrame):
    print("\n--- Starting Advanced Model Training with Hyperparameter Tuning ---")
    
    feature_cols = ['period', 'duration', 'depth', 'stellar_teff', 'stellar_logg', 'impact']
    X = df[feature_cols]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training data size: {len(X_train)} samples")
    print(f"Testing data size:  {len(X_test)} samples")
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
    }

    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

    print("Starting GridSearchCV... This may take a while.")
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
    
    grid_search.fit(X_train, y_train)

    print("\n--- GridSearchCV Complete ---")
    print(f"Best Parameters Found: {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_
    
    print("\n--- Best Model Evaluation ---")
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['False Positive', 'Planet Candidate']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return best_model

def save_model(model) -> None:
    print("\n--- Saving Model ---")
    try:
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
        joblib.dump(model, MODEL_OUTPUT_PATH)
        print(f"Model successfully saved to: {MODEL_OUTPUT_PATH}")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    koi_data = load_and_prepare_koi_data(KOI_FILE_PATH)
    toi_data = load_and_prepare_toi_data(TOI_FILE_PATH)

    if not koi_data.empty or not toi_data.empty:
        combined_data = pd.concat([koi_data, toi_data], ignore_index=True)
        print(f"\nTotal combined training samples: {len(combined_data)}")
        
        trained_model = train_and_evaluate_model(combined_data)
        save_model(trained_model)
    else:
        print("\nCould not load any training data. Exiting.")

