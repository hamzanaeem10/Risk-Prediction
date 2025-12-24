"""
Advanced Training Script for Financial Risk Prediction
Features:
- Domain-specific feature engineering
- Optuna hyperparameter optimization
- Ensemble methods (XGBoost + RandomForest + LightGBM-style voting)
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import mlflow
import mlflow.xgboost
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sqlalchemy import create_engine
import joblib
import os
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Database connection
DB_USER = os.getenv("DB_USER", "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "risk_db")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# MLflow Setup
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
mlflow.set_experiment("financial_risk_prediction")


def load_data_from_warehouse():
    """Load data from PostgreSQL Data Warehouse."""
    print("=" * 60)
    print("Loading data from PostgreSQL Warehouse...")
    print("=" * 60)
    
    try:
        engine = create_engine(DATABASE_URL)
        query = """
        SELECT 
            c.age, c.gender, c.education_level, c.marital_status,
            c.income, c.employment_status, c.years_at_current_job,
            c.number_of_dependents,
            ch.credit_score, ch.payment_history, ch.previous_defaults,
            f.loan_amount, f.loan_purpose, f.debt_to_income_ratio,
            f.assets_value, f.marital_status_change, f.risk_rating
        FROM fact_loans f
        JOIN dim_customers c ON f.customer_id = c.customer_id
        JOIN dim_credit_history ch ON f.credit_history_id = ch.credit_history_id;
        """
        df = pd.read_sql(query, engine)
        print(f"  ✓ Loaded {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error: {e}")
        raise


def create_domain_features(df):
    """Create domain-specific financial risk features."""
    print("\n🔧 Creating Domain-Specific Features...")
    
    # === FINANCIAL RATIOS ===
    df['loan_to_income_ratio'] = df['loan_amount'] / (df['income'] + 1)
    df['asset_coverage_ratio'] = df['assets_value'] / (df['loan_amount'] + 1)
    df['net_worth_estimate'] = df['assets_value'] - df['loan_amount']
    
    # === RISK INDICATORS ===
    df['high_dti_flag'] = (df['debt_to_income_ratio'] > 0.43).astype(int)  # 43% is standard threshold
    df['poor_credit_flag'] = (df['credit_score'] < 580).astype(int)  # Subprime threshold
    df['fair_credit_flag'] = ((df['credit_score'] >= 580) & (df['credit_score'] < 670)).astype(int)
    df['good_credit_flag'] = (df['credit_score'] >= 670).astype(int)
    df['has_previous_defaults'] = (df['previous_defaults'] > 0).astype(int)
    
    # === STABILITY INDICATORS ===
    df['job_stability'] = df['years_at_current_job'] / (df['age'] - 18 + 1)  # % of career at current job
    df['income_per_dependent'] = df['income'] / (df['number_of_dependents'] + 1)
    
    # === CREDIT UTILIZATION PROXY ===
    df['credit_score_normalized'] = (df['credit_score'] - 300) / 550  # Normalize to 0-1
    df['affordability_index'] = (df['income'] - df['loan_amount'] * 0.05) / (df['income'] + 1)  # Assume 5% annual payment
    
    # === INTERACTION FEATURES ===
    df['credit_income_interaction'] = df['credit_score_normalized'] * np.log1p(df['income'])
    df['risk_composite'] = df['high_dti_flag'] + df['poor_credit_flag'] + df['has_previous_defaults']
    
    # === AGE-BASED RISK ===
    df['young_borrower'] = (df['age'] < 25).astype(int)
    df['senior_borrower'] = (df['age'] > 55).astype(int)
    
    print("  ✓ Created 16 domain-specific features")
    return df


def preprocess_data(df):
    """Preprocess data for training."""
    print("\n📊 Preprocessing Data...")
    
    df = df.drop_duplicates()
    df = create_domain_features(df)
    
    target_col = 'risk_rating'
    
    # All numerical features
    numerical_features = [
        'age', 'income', 'credit_score', 'loan_amount', 
        'years_at_current_job', 'debt_to_income_ratio', 
        'assets_value', 'number_of_dependents', 'previous_defaults',
        'marital_status_change',
        # Domain features
        'loan_to_income_ratio', 'asset_coverage_ratio', 'net_worth_estimate',
        'high_dti_flag', 'poor_credit_flag', 'fair_credit_flag', 'good_credit_flag',
        'has_previous_defaults', 'job_stability', 'income_per_dependent',
        'credit_score_normalized', 'affordability_index', 'credit_income_interaction',
        'risk_composite', 'young_borrower', 'senior_borrower'
    ]
    
    categorical_features = [
        'gender', 'education_level', 'marital_status', 
        'loan_purpose', 'employment_status', 'payment_history'
    ]
    
    # Handle missing/infinite values
    for col in numerical_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].median())
    
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # One-hot encode
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # Encode target
    label_encoder = LabelEncoder()
    df_encoded[target_col] = label_encoder.fit_transform(df_encoded[target_col])
    
    feature_cols = [col for col in df_encoded.columns if col != target_col]
    X = df_encoded[feature_cols].copy()
    y = df_encoded[target_col]
    
    # Scale
    scaler = StandardScaler()
    cols_to_scale = [col for col in numerical_features if col in X.columns]
    X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
    
    print(f"  ✓ Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"  ✓ Classes: {dict(zip(label_encoder.classes_, y.value_counts().sort_index().tolist()))}")
    
    return X, y, label_encoder, scaler, list(X.columns)


def optuna_objective(trial, X_train, y_train, X_val, y_val, feature_cols):
    """Optuna objective function for XGBoost hyperparameter tuning."""
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
        'eval_metric': 'mlogloss',
        'seed': 42
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
    
    model = xgb.train(
        params, dtrain, 
        num_boost_round=trial.suggest_int('n_estimators', 100, 500),
        evals=[(dval, 'val')],
        early_stopping_rounds=30,
        verbose_eval=False
    )
    
    y_pred = np.argmax(model.predict(dval), axis=1)
    return f1_score(y_val, y_pred, average='weighted')


def train_advanced():
    """Train with Optuna optimization and ensemble methods."""
    df = load_data_from_warehouse()
    X, y, label_encoder, scaler, feature_cols = preprocess_data(df)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # SMOTE
    print("\n⚖️  Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"  ✓ Resampled: {len(X_train_res)} samples")
    
    # Further split for Optuna validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_res, y_train_res, test_size=0.2, random_state=42
    )
    
    with mlflow.start_run(run_name="Advanced_Optimized"):
        # === OPTUNA HYPERPARAMETER TUNING ===
        print("\n🔍 Optuna Hyperparameter Optimization (50 trials)...")
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: optuna_objective(trial, X_tr, y_tr, X_val, y_val, feature_cols),
            n_trials=50,
            show_progress_bar=True
        )
        
        best_params = study.best_params
        print(f"\n  ✓ Best F1 Score: {study.best_value:.4f}")
        print(f"  ✓ Best Params: {best_params}")
        
        mlflow.log_params({f"optuna_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("optuna_best_f1", study.best_value)
        
        # === TRAIN OPTIMIZED XGBOOST ===
        print("\n🚀 Training Optimized XGBoost...")
        xgb_params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': best_params['max_depth'],
            'eta': best_params['eta'],
            'subsample': best_params['subsample'],
            'colsample_bytree': best_params['colsample_bytree'],
            'min_child_weight': best_params['min_child_weight'],
            'gamma': best_params['gamma'],
            'reg_alpha': best_params['reg_alpha'],
            'reg_lambda': best_params['reg_lambda'],
            'eval_metric': 'mlogloss',
            'seed': 42
        }
        
        dtrain = xgb.DMatrix(X_train_res, label=y_train_res, feature_names=feature_cols)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)
        
        xgb_model = xgb.train(
            xgb_params, dtrain,
            num_boost_round=best_params['n_estimators'],
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        y_pred_xgb = np.argmax(xgb_model.predict(dtest), axis=1)
        acc_xgb = accuracy_score(y_test, y_pred_xgb)
        f1_xgb = f1_score(y_test, y_pred_xgb, average='weighted')
        
        print("\n  XGBoost Results:")
        print(f"    Accuracy: {acc_xgb:.4f}")
        print(f"    F1 (Weighted): {f1_xgb:.4f}")
        
        # === ENSEMBLE: RandomForest ===
        print("\n🌲 Training RandomForest for Ensemble...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=best_params['max_depth'],
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_res, y_train_res)
        y_pred_rf = rf_model.predict(X_test)
        f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
        print(f"    RandomForest F1: {f1_rf:.4f}")
        
        # === ENSEMBLE: GradientBoosting ===
        print("🌳 Training GradientBoosting for Ensemble...")
        gb_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=min(best_params['max_depth'], 8),
            learning_rate=0.1,
            random_state=42
        )
        gb_model.fit(X_train_res, y_train_res)
        y_pred_gb = gb_model.predict(X_test)
        f1_gb = f1_score(y_test, y_pred_gb, average='weighted')
        print(f"    GradientBoosting F1: {f1_gb:.4f}")
        
        # === SOFT VOTING ENSEMBLE ===
        print("\n🗳️  Creating Voting Ensemble...")
        # Average probabilities from all models
        xgb_probs = xgb_model.predict(dtest)
        rf_probs = rf_model.predict_proba(X_test)
        gb_probs = gb_model.predict_proba(X_test)
        
        ensemble_probs = (xgb_probs + rf_probs + gb_probs) / 3
        y_pred_ensemble = np.argmax(ensemble_probs, axis=1)
        
        acc_ensemble = accuracy_score(y_test, y_pred_ensemble)
        f1_ensemble = f1_score(y_test, y_pred_ensemble, average='weighted')
        f1_macro_ensemble = f1_score(y_test, y_pred_ensemble, average='macro')
        
        print(f"\n{'='*60}")
        print("📊 FINAL RESULTS - ENSEMBLE MODEL")
        print(f"{'='*60}")
        print(f"  Accuracy:     {acc_ensemble:.4f}")
        print(f"  F1 (Weighted): {f1_ensemble:.4f}")
        print(f"  F1 (Macro):    {f1_macro_ensemble:.4f}")
        print(f"\n{classification_report(y_test, y_pred_ensemble, target_names=label_encoder.classes_)}")
        
        # Log metrics
        mlflow.log_metric("xgb_accuracy", acc_xgb)
        mlflow.log_metric("xgb_f1_weighted", f1_xgb)
        mlflow.log_metric("rf_f1_weighted", f1_rf)
        mlflow.log_metric("gb_f1_weighted", f1_gb)
        mlflow.log_metric("ensemble_accuracy", acc_ensemble)
        mlflow.log_metric("ensemble_f1_weighted", f1_ensemble)
        mlflow.log_metric("ensemble_f1_macro", f1_macro_ensemble)
        
        # Save models
        model_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(model_dir, exist_ok=True)
        
        xgb_model.save_model(os.path.join(model_dir, "xgboost_model.json"))
        joblib.dump(rf_model, os.path.join(model_dir, "rf_model.pkl"))
        joblib.dump(gb_model, os.path.join(model_dir, "gb_model.pkl"))
        joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))
        joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
        joblib.dump(feature_cols, os.path.join(model_dir, "feature_cols.pkl"))
        
        mlflow.xgboost.log_model(xgb_model, "xgboost_model")
        mlflow.sklearn.log_model(rf_model, "rf_model")
        
        print(f"\n✅ Models saved to {model_dir}")
        print(f"🔗 View run: {mlflow.active_run().info.artifact_uri}")


if __name__ == "__main__":
    train_advanced()
