import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, List, Any

# Machine Learning Imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# ----------------------------
# Paths & Constants
# ----------------------------
ART_DIR = "artifacts"
if not os.path.exists(ART_DIR):
    os.makedirs(ART_DIR)

RESULTS_PATH = os.path.join(ART_DIR, "results.csv")
MODEL_PATH = os.path.join(ART_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(ART_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(ART_DIR, "feature_names.json")
DEFAULT_MERGED_DATA_PATH = "price.csv" 

# Column names
COL_DATE, COL_PRICE, COL_PROD, COL_EXPORT, COL_PRECIP = "Date", "Price", "Index Production", "Export Number (in Tonnes)", "Precip"
OPTIONAL_COLS = ["Temp", "Humidity", "USD"]
MAX_DATE_STR = "2022-05-31"
MAX_DATE = pd.to_datetime(MAX_DATE_STR)

# ----------------------------
# Streamlit Config
# ----------------------------
st.set_page_config(page_title="Palm Oil Price Prediction App", layout="wide")
st.title("🌴 Palm Oil Price Forecasting Dashboard")

# ----------------------------
# Shared Helpers
# ----------------------------
@st.cache_data
def load_merged_dataset(path):
    df = pd.read_csv(path)
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    df = df.dropna(subset=[COL_DATE]).sort_values(COL_DATE).reset_index(drop=True)
    df = df[df[COL_DATE] <= MAX_DATE].copy()
    for c in [COL_PRICE, COL_PROD, COL_EXPORT, COL_PRECIP] + OPTIONAL_COLS:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def get_best_row(df):
    metric = "RMSE" if "RMSE" in df.columns else "R2"
    return df.sort_values(metric, ascending=(metric == "RMSE")).iloc[0]

# ----------------------------
# Tabs Definition
# ----------------------------
tab_dash, tab_train, tab_compare, tab_pred = st.tabs(["📊 Dashboard", "⚙️ Training", "🏆 Comparison", "🧠 Prediction"])

# ============================================================
# TAB 1: DASHBOARD
# ============================================================
with tab_dash:
    st.subheader("Market Insights")
    if os.path.exists(DEFAULT_MERGED_DATA_PATH):
        merged_df = load_merged_dataset(DEFAULT_MERGED_DATA_PATH)
        # Add your plots here (daily_price_trend_plot, etc.)
        st.line_chart(merged_df.set_index(COL_DATE)[COL_PRICE])
    else:
        st.warning("Please upload or ensure data exists at 'data/final_merged_palm_oil_dataset.csv'")

# ============================================================
# TAB 2: TRAINING (Your ML Logic)
# ============================================================
with tab_train:
    st.subheader("Model Training & Hyperparameter Tuning")
    
    if os.path.exists(DEFAULT_MERGED_DATA_PATH):
        train_df = load_merged_dataset(DEFAULT_MERGED_DATA_PATH).dropna()
        features = [c for c in train_df.columns if c not in [COL_DATE, COL_PRICE]]
        X, y = train_df[features], train_df[COL_PRICE]
        
        split_ratio = st.slider("Train Split %", 60, 90, 80)
        if st.button("🚀 Run Training Pipeline"):
            with st.spinner("Training models..."):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-split_ratio)/100, random_state=42)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Model Logic
                models = {"XGBoost": XGBRegressor(), "Random Forest": RandomForestRegressor()}
                results = []
                
                # Example for XGBoost (Simplified)
                m = models["XGBoost"]
                m.fit(X_train_scaled, y_train)
                preds = m.predict(X_test_scaled)
                
                res = {"Model": "XGBoost", "R2": r2_score(y_test, preds), "RMSE": np.sqrt(mean_squared_error(y_test, preds))}
                results.append(res)
                
                # Save Artifacts
                joblib.dump(m, MODEL_PATH)
                joblib.dump(scaler, SCALER_PATH)
                pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
                with open(FEATURES_PATH, "w") as f: json.dump(list(features), f)
                
                st.success("Training Complete! Artifacts saved to /artifacts")
                st.table(results)

# ============================================================
# TAB 3 & 4: (Use original logic to load from RESULTS_PATH and MODEL_PATH)
# ============================================================
# ============================================================
# TAB 3: PREDICTION
# ============================================================
with tab_pred:
    st.subheader("🧠 Palm Oil Price Estimation")

    st.markdown(
        f"""
        <div style="padding:12px 14px; border-radius:14px; background:#f3f6ff; border:1px solid #dbe4ff;">
          <div style="font-size:20px; font-weight:850;">Model used: {best_model_name}</div>
          <div style="font-size:12.5px; opacity:0.8;">
            Defaults are derived from dataset mean (restricted to up to {MAX_DATE_STR}).
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

    if not os.path.exists(DEFAULT_MERGED_DATA_PATH):
        stop_with_error(f"Missing merged dataset file: {DEFAULT_MERGED_DATA_PATH}")

    defaults = compute_feature_defaults(DEFAULT_MERGED_DATA_PATH, feature_names)

    st.markdown("### Set Parameters for Single Prediction")
    st.caption(
        "All parameters are pre-filled using mean values from the dataset (up to 31-05-2022). "
        "Adjust any value to test different conditions."
    )

    inputs: Dict[str, Any] = {}
    cols = st.columns(2, gap="large")

    for i, feat in enumerate(feature_names):
        if feat.strip().lower() == "year":
            continue

        default_val = defaults.get(feat, 0.0)

        with cols[i % 2]:
            if feat == COL_EXPORT:
                inputs[feat] = st.number_input(
                    feat,
                    value=int(round(default_val)),
                    step=1,
                    format="%d",
                    key=f"single_{feat}"
                )
            else:
                inputs[feat] = st.number_input(
                    feat,
                    value=float(default_val),
                    step=0.1,
                    key=f"single_{feat}"
                )

    year_feat = next((f for f in feature_names if f.strip().lower() == "year"), None)
    if year_feat is not None:
        inputs[year_feat] = 2022

    st.write("")

    c1, c2 = st.columns([1, 1])
    with c1:
        do_pred = st.button("Predict Price ✅")
    with c2:
        reset = st.button("Reset output")

    if reset:
        st.session_state.single_pred_value = None
        st.session_state.single_pred_inputs = None

    if do_pred:
        try:
            pred = predict_one(inputs)
            st.session_state.single_pred_value = pred
            st.session_state.single_pred_inputs = inputs
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.divider()
    st.markdown("### Outcome")

    if st.session_state.single_pred_value is None:
        st.info("No prediction yet. Adjust values and click Predict.")
    else:
        st.metric("Predicted Palm Oil Price per Tonne", f"RM {st.session_state.single_pred_value:,.2f}")
        with st.expander("Show inputs used"):
            st.json(st.session_state.single_pred_inputs)
