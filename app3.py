import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Manufacturing Price Prediction",
    layout="wide"
)

st.title("Manufacturing Price Prediction Dashboard")
st.markdown(
    """
    This application predicts manufacturing-related prices using 
    multiple machine learning regression models.
    """
)

# ============================================================
# Load Data
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("price.csv")
    return df

df = load_data()

# ============================================================
# Date Handling (if exists)
# ============================================================
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

# ============================================================
# Sidebar – User Controls
# ============================================================
st.sidebar.header("Configuration")

# Model selection
model_name = st.sidebar.selectbox(
    "Select Machine Learning Model",
    [
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost"
    ]
)

# Date / Month filtering
if "date" in df.columns:
    min_date = df["date"].min()
    max_date = df["date"].max()

    selected_date = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if len(selected_date) == 2:
        df = df[
            (df["date"] >= pd.to_datetime(selected_date[0])) &
            (df["date"] <= pd.to_datetime(selected_date[1]))
        ]

# ============================================================
# Feature Engineering Selection
# ============================================================
target_column = "price"

numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features.remove(target_column)

selected_features = st.sidebar.multiselect(
    "Select Features for Prediction",
    options=numeric_features,
    default=numeric_features
)

# ============================================================
# Data Preview
# ============================================================
with st.expander("Preview Dataset"):
    st.dataframe(df.head())

# ============================================================
# Train-Test Split
# ============================================================
X = df[selected_features]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# Model Initialization
# ============================================================
if model_name == "Decision Tree":
    model = DecisionTreeRegressor(
        max_depth=5,
        random_state=42
    )

elif model_name == "Random Forest":
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

elif model_name == "Gradient Boosting":
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

elif model_name == "XGBoost":
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

# ============================================================
# Model Training
# ============================================================
model.fit(X_train, y_train)

# ============================================================
# Evaluation
# ============================================================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# ============================================================
# Results Display
# ============================================================
st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{mae:.2f}")
col2.metric("RMSE", f"{rmse:.2f}")
col3.metric("R² Score", f"{r2:.3f}")

# ============================================================
# Prediction Section
# ============================================================
st.subheader("Make a Prediction")

input_data = {}

for feature in selected_features:
    input_data[feature] = st.number_input(
        f"{feature}",
        value=float(df[feature].mean())
    )

input_df = pd.DataFrame([input_data])

if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Price: {prediction:.2f}")
