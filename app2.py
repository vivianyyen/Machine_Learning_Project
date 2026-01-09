# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Page configuration
st.set_page_config(
    page_title="Palm Oil Price Predictor",
    page_icon="🌴",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #3CB371;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .stButton>button {
        background-color: #2E8B57;
        color: white;
        font-weight: bold;
    }
    .best-model {
        border: 3px solid #FFD700;
        padding: 10px;
        border-radius: 10px;
        background-color: #FFF8DC;
    }
    .train-metrics {
        background-color: #f0f8ff;
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid #3CB371;
    }
    .test-metrics {
        background-color: #fff0f0;
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid #FF6B6B;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🌴 Palm Oil Price Prediction System</h1>', unsafe_allow_html=True)
st.markdown("""
This application predicts palm oil prices using machine learning models with hyperparameter tuning.
The system integrates weather data, production indices, exchange rates, and export numbers to provide accurate predictions.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Data Overview", "Model Predictions", "Results Comparison", "Hyperparameter Tuning"])

# Cache data loading
@st.cache_data
def load_data():
    """Load and prepare data"""
    # Load your data - adjust filename as needed
    df = pd.read_csv("price.csv")
    # Ensure Date column is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Add year and month columns if not present
    if 'Date' in df.columns:
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
    
    # Fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
        
    return df

def get_hyperparameter_grids():
    """Define hyperparameter grids for all models"""
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.6, 0.8, 1.0]
        },
        'SVR': {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.5, 1.0],
            'kernel': ['linear', 'rbf', 'poly']
        },
        'Decision Tree': {
            'max_depth': [3, 5, 7, 10, 15, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
            'splitter': ['best', 'random']
        }
    }
    return param_grids

@st.cache_resource
def train_models_with_tuning(X_train, y_train, X_test, y_test, tuning_method='grid', cv_folds=3):
    """Train models with hyperparameter tuning and calculate metrics"""
    models = {}
    best_params = {}
    train_metrics = {}
    test_metrics = {}
    
    # Get hyperparameter grids
    param_grids = get_hyperparameter_grids()
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Random Forest Regressor
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    if tuning_method == 'grid':
        rf_search = GridSearchCV(
            rf, 
            param_grids['Random Forest'], 
            cv=cv_folds, 
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
    else:
        rf_search = RandomizedSearchCV(
            rf,
            param_grids['Random Forest'],
            n_iter=20,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
    
    rf_search.fit(X_train_scaled, y_train)
    models['Random Forest'] = rf_search.best_estimator_
    best_params['Random Forest'] = rf_search.best_params_
    
    # Calculate metrics
    y_train_pred = rf_search.best_estimator_.predict(X_train_scaled)
    y_test_pred = rf_search.best_estimator_.predict(X_test_scaled)
    
    train_metrics['Random Forest'] = {
        'R²': r2_score(y_train, y_train_pred),
        'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'MAE': mean_absolute_error(y_train, y_train_pred)
    }
    
    test_metrics['Random Forest'] = {
        'R²': r2_score(y_test, y_test_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'MAE': mean_absolute_error(y_test, y_test_pred)
    }
    
    # 2. XGBoost Regressor
    xgb = XGBRegressor(random_state=42, n_jobs=-1)
    
    if tuning_method == 'grid':
        xgb_search = GridSearchCV(
            xgb,
            param_grids['XGBoost'],
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
    else:
        xgb_search = RandomizedSearchCV(
            xgb,
            param_grids['XGBoost'],
            n_iter=20,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
    
    xgb_search.fit(X_train_scaled, y_train)
    models['XGBoost'] = xgb_search.best_estimator_
    best_params['XGBoost'] = xgb_search.best_params_
    
    # Calculate metrics
    y_train_pred = xgb_search.best_estimator_.predict(X_train_scaled)
    y_test_pred = xgb_search.best_estimator_.predict(X_test_scaled)
    
    train_metrics['XGBoost'] = {
        'R²': r2_score(y_train, y_train_pred),
        'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'MAE': mean_absolute_error(y_train, y_train_pred)
    }
    
    test_metrics['XGBoost'] = {
        'R²': r2_score(y_test, y_test_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'MAE': mean_absolute_error(y_test, y_test_pred)
    }
    
    # 3. Gradient Boosting Regressor
    gbr = GradientBoostingRegressor(random_state=42)
    
    if tuning_method == 'grid':
        gbr_search = GridSearchCV(
            gbr,
            param_grids['Gradient Boosting'],
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
    else:
        gbr_search = RandomizedSearchCV(
            gbr,
            param_grids['Gradient Boosting'],
            n_iter=20,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
    
    gbr_search.fit(X_train_scaled, y_train)
    models['Gradient Boosting'] = gbr_search.best_estimator_
    best_params['Gradient Boosting'] = gbr_search.best_params_
    
    # Calculate metrics
    y_train_pred = gbr_search.best_estimator_.predict(X_train_scaled)
    y_test_pred = gbr_search.best_estimator_.predict(X_test_scaled)
    
    train_metrics['Gradient Boosting'] = {
        'R²': r2_score(y_train, y_train_pred),
        'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'MAE': mean_absolute_error(y_train, y_train_pred)
    }
    
    test_metrics['Gradient Boosting'] = {
        'R²': r2_score(y_test, y_test_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'MAE': mean_absolute_error(y_test, y_test_pred)
    }
    
    # 4. SVR (Support Vector Regression)
    svr = SVR()
    
    if tuning_method == 'grid':
        svr_search = GridSearchCV(
            svr,
            param_grids['SVR'],
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
    else:
        svr_search = RandomizedSearchCV(
            svr,
            param_grids['SVR'],
            n_iter=20,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
    
    svr_search.fit(X_train_scaled, y_train)
    models['SVR'] = svr_search.best_estimator_
    best_params['SVR'] = svr_search.best_params_
    
    # Calculate metrics
    y_train_pred = svr_search.best_estimator_.predict(X_train_scaled)
    y_test_pred = svr_search.best_estimator_.predict(X_test_scaled)
    
    train_metrics['SVR'] = {
        'R²': r2_score(y_train, y_train_pred),
        'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'MAE': mean_absolute_error(y_train, y_train_pred)
    }
    
    test_metrics['SVR'] = {
        'R²': r2_score(y_test, y_test_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'MAE': mean_absolute_error(y_test, y_test_pred)
    }
    
    # 5. Decision Tree Regressor
    dt = DecisionTreeRegressor(random_state=42)
    
    if tuning_method == 'grid':
        dt_search = GridSearchCV(
            dt,
            param_grids['Decision Tree'],
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
    else:
        dt_search = RandomizedSearchCV(
            dt,
            param_grids['Decision Tree'],
            n_iter=20,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
    
    dt_search.fit(X_train_scaled, y_train)
    models['Decision Tree'] = dt_search.best_estimator_
    best_params['Decision Tree'] = dt_search.best_params_
    
    # Calculate metrics
    y_train_pred = dt_search.best_estimator_.predict(X_train_scaled)
    y_test_pred = dt_search.best_estimator_.predict(X_test_scaled)
    
    train_metrics['Decision Tree'] = {
        'R²': r2_score(y_train, y_train_pred),
        'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'MAE': mean_absolute_error(y_train, y_train_pred)
    }
    
    test_metrics['Decision Tree'] = {
        'R²': r2_score(y_test, y_test_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'MAE': mean_absolute_error(y_test, y_test_pred)
    }
    
    return models, best_params, train_metrics, test_metrics, scaler

@st.cache_resource
def train_models_basic(X_train, y_train, X_test, y_test):
    """Train models without hyperparameter tuning and calculate metrics"""
    models = {}
    train_metrics = {}
    test_metrics = {}
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Random Forest Regressor
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    start_time = time.time
    rf.fit(X_train_scaled, y_train)
    models['Random Forest'] = rf
    end_time = time.time
    training_time_rf = end_time - start_time
    # Calculate metrics
    y_train_pred = rf.predict(X_train_scaled)
    y_test_pred = rf.predict(X_test_scaled)
    
    
    test_metrics['Random Forest'] = {
        'R²': r2_score(y_test, y_test_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'MAE': mean_absolute_error(y_test, y_test_pred)
    }
    
    # 2. XGBoost Regressor
    xgb = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    start_time = time.time
    xgb.fit(X_train_scaled, y_train)
    models['XGBoost'] = xgb
    end_time = time.time
    training_time_xgb = end_time - start_time
    
    # Calculate metrics
    y_train_pred = xgb.predict(X_train_scaled)
    y_test_pred = xgb.predict(X_test_scaled)
    
    
    test_metrics['XGBoost'] = {
        'R²': r2_score(y_test, y_test_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'MAE': mean_absolute_error(y_test, y_test_pred)
    }
    
    # 3. Gradient Boosting Regressor
    gbr = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    start_time = time.time
    gbr.fit(X_train_scaled, y_train)
    models['Gradient Boosting'] = gbr
    end_time = time.time
    training_time_gbr= end_time - start_time

    
    # Calculate metrics
    y_train_pred = gbr.predict(X_train_scaled)
    y_test_pred = gbr.predict(X_test_scaled)
    
    test_metrics['Gradient Boosting'] = {
        'R²': r2_score(y_test, y_test_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'MAE': mean_absolute_error(y_test, y_test_pred)
    }
    
    # 4. SVR (Support Vector Regression)
    svr = SVR(kernel='rbf', C=10, epsilon=0.1)
    start_time = time.time
    svr.fit(X_train_scaled, y_train)
    models['SVR'] = svr
    end_time = time.time
    training_time_svr = end_time - start_time
    
    # Calculate metrics
    y_train_pred = svr.predict(X_train_scaled)
    y_test_pred = svr.predict(X_test_scaled)
    
    
    test_metrics['SVR'] = {
        'R²': r2_score(y_test, y_test_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'MAE': mean_absolute_error(y_test, y_test_pred)
    }
    
    # 5. Decision Tree Regressor
    dt = DecisionTreeRegressor(
        max_depth=5,
        min_samples_leaf=5,
        random_state=42
    )
    start_time = time.time
    dt.fit(X_train_scaled, y_train)
    models['Decision Tree'] = dt
    end_time = time.time
    training_time_dt = end_time - start_time
    
    # Calculate metrics
    y_train_pred = dt.predict(X_train_scaled)
    y_test_pred = dt.predict(X_test_scaled)
    
    test_metrics['Decision Tree'] = {
        'R²': r2_score(y_test, y_test_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'MAE': mean_absolute_error(y_test, y_test_pred)
    }
    
    return models, train_metrics, test_metrics, scaler

# Load data
df = load_data()

if page == "Data Overview":
    st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Data Sample")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.subheader("📊 Data Information")
        buffer = []
        buffer.append(f"**Total Rows:** {df.shape[0]}\n\n")
        buffer.append(f"**Total Columns:** {df.shape[1]}\n\n")
        if 'Date' in df.columns:
            buffer.append(f"**Date Range:** {df['Date'].min().date()} to {df['Date'].max().date()}\n")
        if 'Price' in df.columns:
            buffer.append(f"**Average Price:** ${df['Price'].mean():.2f}\n\n")
            buffer.append(f"**Price Range:** ${df['Price'].min():.2f} - ${df['Price'].max():.2f}\n\n")
        
        st.markdown("\n".join(buffer))
    
    # Line chart of price over time
    if 'Date' in df.columns and 'Price' in df.columns:
        st.subheader("📈 Price Distribution Over Time")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['Date'], df['Price'], linewidth=2, color='#2E8B57')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.set_title('Palm Oil Price Trend')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Date or Price column not found in dataset.")
    
    # Basic statistics
    st.subheader("📊 Statistical Summary")
    if 'Price' in df.columns:
        price_stats = df['Price'].describe()
        cols = st.columns(4)
        with cols[0]:
            st.metric("Mean", f"${price_stats['mean']:.2f}")
        with cols[1]:
            st.metric("Std Dev", f"${price_stats['std']:.2f}")
        with cols[2]:
            st.metric("Min", f"${price_stats['min']:.2f}")
        with cols[3]:
            st.metric("Max", f"${price_stats['max']:.2f}")

elif page == "Model Predictions":
    st.markdown('<h2 class="sub-header">Model Training & Prediction</h2>', unsafe_allow_html=True)
    
    if 'Price' not in df.columns:
        st.error("'Price' column not found in dataset. Cannot proceed with modeling.")
        st.info("Please ensure your dataset contains a 'Price' column.")
    else:
        # Select available features
        possible_features = ['Solarradiation', 'Solarenergy', 'Uvindex', 
                           'Index Production', 'Export Number (in Tonnes)', 
                           'USD', 'Year', 'Month']
        
        available_features = [f for f in possible_features if f in df.columns]
        
        if len(available_features) < 3:
            st.error(f"Not enough features available for modeling. Found: {available_features}")
        else:
            X = df[available_features]
            y = df['Price']
            
            # Handle any missing values
            X = X.fillna(X.median())
            y = y.fillna(y.median())
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Training options
            st.sidebar.subheader("Training Options")
            use_hyperparameter_tuning = st.sidebar.checkbox("Use Hyperparameter Tuning", value=True)
            tuning_method = st.sidebar.selectbox("Tuning Method", ["grid", "random"], index=0)
            cv_folds = st.sidebar.slider("CV Folds", min_value=3, max_value=10, value=3)
            
            # Feature selection using RFE
            with st.expander("🎯 Feature Selection Details"):
                estimator = LinearRegression()
                n_features = min(5, len(available_features))
                rfe = RFE(estimator=estimator, n_features_to_select=n_features)
                
                # Scale for RFE
                scaler_rfe = StandardScaler()
                X_train_scaled_rfe = scaler_rfe.fit_transform(X_train)
                
                rfe.fit(X_train_scaled_rfe, y_train)
                
                selected_features = X_train.columns[rfe.support_].tolist()
                st.write(f"**Selected {n_features} features:** {', '.join(selected_features)}")
            
            # Train models button
            if st.button("🚀 Train All Models", type="primary"):
                with st.spinner("Training models... This may take a few minutes."):
                    if use_hyperparameter_tuning:
                        models, best_params, train_metrics, test_metrics, scaler = train_models_with_tuning(
                            X_train, y_train, X_test, y_test, tuning_method, cv_folds
                        )
                        tuning_status = "with Hyperparameter Tuning"
                    else:
                        models, train_metrics, test_metrics, scaler = train_models_basic(
                            X_train, y_train, X_test, y_test
                        )
                        tuning_status = "with Default Parameters"
                        best_params = {}
                
                if models:
                    st.success(f"✅ All models trained successfully {tuning_status}!")
                    
                    # Display best parameters if tuning was used
                    if use_hyperparameter_tuning and best_params:
                        with st.expander("📋 Best Hyperparameters"):
                            for model_name, params in best_params.items():
                                st.write(f"**{model_name}:**")
                                st.json(params)
                               
                    # Store metrics in session state for comparison page
                    st.session_state['train_metrics'] = train_metrics
                    st.session_state['test_metrics'] = test_metrics
                    st.session_state['models'] = models
                    st.session_state['scaler'] = scaler
                    st.session_state['X_train'] = X_train
                    st.session_state['y_train'] = y_train
                    st.session_state['X_test'] = X_test
                    st.session_state['y_test'] = y_test
                    st.session_state['available_features'] = available_features
                    
                    # Select a model to view predictions
                    st.subheader("📊 Model Predictions")
                    
                    selected_model = st.selectbox(
                        "Choose a model to view detailed predictions:",
                        list(models.keys())
                    )
                    
                    model = models[selected_model]
                    
                    # Make predictions for the selected model
                    X_test_scaled = scaler.transform(X_test)
                    y_pred = model.predict(X_test_scaled)
                    
                    # Display train vs test metrics comparison
                    st.subheader("📈 Train vs Test Performance")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("### R² Score")
                        st.metric("Train", f"{train_metrics[selected_model]['R²']:.4f}")
                        st.metric("Test", f"{test_metrics[selected_model]['R²']:.4f}")
                        diff_r2 = test_metrics[selected_model]['R²'] - train_metrics[selected_model]['R²']
                        st.caption(f"Difference: {diff_r2:+.4f}")
                    
                    with col2:
                        st.markdown("### RMSE")
                        st.metric("Train", f"{train_metrics[selected_model]['RMSE']:.2f}")
                        st.metric("Test", f"{test_metrics[selected_model]['RMSE']:.2f}")
                        diff_rmse = test_metrics[selected_model]['RMSE'] - train_metrics[selected_model]['RMSE']
                        st.caption(f"Difference: {diff_rmse:+.2f}")
                    
                    with col3:
                        st.markdown("### MAE")
                        st.metric("Train", f"{train_metrics[selected_model]['MAE']:.2f}")
                        st.metric("Test", f"{test_metrics[selected_model]['MAE']:.2f}")
                        diff_mae = test_metrics[selected_model]['MAE'] - train_metrics[selected_model]['MAE']
                        st.caption(f"Difference: {diff_mae:+.2f}")
                    
                    # Overfitting/Underfitting analysis
                    st.subheader("🔍 Overfitting Analysis")
                    r2_gap = abs(train_metrics[selected_model]['R²'] - test_metrics[selected_model]['R²'])
                    
                    if r2_gap > 0.1:
                        st.warning(f"⚠️ Potential overfitting detected! R² gap: {r2_gap:.3f}")
                        st.info("The model performs much better on training data than test data. Consider:")
                        st.write("1. Reducing model complexity")
                        st.write("2. Adding regularization")
                        st.write("3. Getting more training data")
                    elif r2_gap < 0.05:
                        st.success(f"✅ Good generalization! R² gap: {r2_gap:.3f}")
                        st.info("The model performs similarly on training and test data.")
                    else:
                        st.info(f"📊 Moderate generalization. R² gap: {r2_gap:.3f}")
                    
                    # Plot predictions vs actual
                    st.subheader("📊 Prediction Visualization")
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                    
                    # Scatter plot
                    ax1.scatter(y_test, y_pred, alpha=0.5, color='#2E8B57', s=30)
                    ax1.plot([y_test.min(), y_test.max()], 
                            [y_test.min(), y_test.max()], 
                            'r--', lw=2, label='Perfect Prediction')
                    ax1.set_xlabel('Actual Price ($)')
                    ax1.set_ylabel('Predicted Price ($)')
                    ax1.set_title(f'{selected_model}: Test Set Predictions')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Residual plot
                    residuals = y_test - y_pred
                    ax2.scatter(y_pred, residuals, alpha=0.5, color='#FF6B6B', s=30)
                    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
                    ax2.set_xlabel('Predicted Price ($)')
                    ax2.set_ylabel('Residuals')
                    ax2.set_title(f'{selected_model}: Residual Plot')
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Show sample predictions
                    st.subheader("📋 Sample Test Predictions (First 20 samples)")
                    results_df = pd.DataFrame({
                        'Actual Price': y_test.values[:20],
                        'Predicted Price': y_pred[:20],
                        'Difference': y_pred[:20] - y_test.values[:20],
                        'Error %': abs((y_pred[:20] - y_test.values[:20]) / y_test.values[:20] * 100)
                    })
                    st.dataframe(results_df.style.format({
                        'Actual Price': '${:.2f}',
                        'Predicted Price': '${:.2f}',
                        'Difference': '${:.2f}',
                        'Error %': '{:.2f}%'
                    }), use_container_width=True)
                    
                    # Feature importance for tree-based models
                    if hasattr(model, 'feature_importances_'):
                        st.subheader("🎯 Feature Importance")
                        importance = pd.DataFrame({
                            'Feature': available_features,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.barh(importance['Feature'], importance['Importance'], color='#2E8B57')
                        ax.set_xlabel('Importance Score')
                        ax.set_title(f'{selected_model}: Feature Importance')
                        ax.invert_yaxis()
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        st.dataframe(importance.style.format({'Importance': '{:.4f}'}), use_container_width=True)
                    
                    # Interactive prediction
                    st.subheader("🎮 Make a Custom Prediction")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        solar_rad = st.slider("Solar Radiation", 
                                             min_value=0.0, 
                                             max_value=500.0, 
                                             value=200.0, 
                                             step=10.0)
                        solar_energy = st.slider("Solar Energy", 
                                                min_value=0.0, 
                                                max_value=30.0, 
                                                value=15.0, 
                                                step=0.5)
                        uv_index = st.slider("UV Index", 
                                            min_value=0.0, 
                                            max_value=15.0, 
                                            value=5.0, 
                                            step=0.5)
                    
                    with col2:
                        production_idx = st.slider("Production Index", 
                                                  min_value=50.0, 
                                                  max_value=150.0, 
                                                  value=100.0, 
                                                  step=5.0)
                        year = st.selectbox("Year", [2020, 2021, 2022], index=1)
                        month = st.selectbox("Month", range(1, 13), index=5)
                    
                    if st.button("🔮 Predict Price with All Models", type="secondary"):
                        # Create input array with all available features
                        input_dict = {}
                        for feature in available_features:
                            if feature == 'Solarradiation':
                                input_dict[feature] = solar_rad
                            elif feature == 'Solarenergy':
                                input_dict[feature] = solar_energy
                            elif feature == 'Uvindex':
                                input_dict[feature] = uv_index
                            elif feature == 'Index Production':
                                input_dict[feature] = production_idx
                            elif feature == 'Year':
                                input_dict[feature] = year
                            elif feature == 'Month':
                                input_dict[feature] = month
                            else:
                                # Use median for other features
                                input_dict[feature] = X_train[feature].median()
                        
                        # Create input array in correct order
                        input_array = np.array([[input_dict[feat] for feat in available_features]])
                        
                        # Scale input
                        input_scaled = scaler.transform(input_array)
                        
                        # Get predictions from all models
                        st.subheader("📈 Prediction Results from All Models")
                        
                        predictions = {}
                        for name, model in models.items():
                            pred = model.predict(input_scaled)[0]
                            predictions[name] = pred
                        
                        # Display results in columns
                        cols = st.columns(len(predictions))
                        model_names = list(predictions.keys())
                        
                        for idx, model_name in enumerate(model_names):
                            with cols[idx]:
                                st.metric(
                                    label=model_name,
                                    value=f"${predictions[model_name]:.2f}",
                                    delta="Predicted"
                                )
                        
                        # Show comparison table
                        st.subheader("📊 Model Comparison")
                        comparison_df = pd.DataFrame({
                            'Model': model_names,
                            'Predicted Price': [predictions[m] for m in model_names]
                        }).sort_values('Predicted Price', ascending=False)
                        
                        st.dataframe(comparison_df.style.format({'Predicted Price': '${:.2f}'}), 
                                   use_container_width=True)
                        
                        best_model = comparison_df.iloc[0]['Model']
                        best_price = comparison_df.iloc[0]['Predicted Price']
                        
                        st.success(f"💰 **Highest predicted price**: **{best_model}** at **${best_price:.2f}**")
                else:
                    st.error("Failed to train models. Please check your data.")
            else:
                st.info("Click '🚀 Train All Models' button to start training and see results.")

elif page == "Results Comparison":
    st.markdown('<h2 class="sub-header">Model Performance Comparison</h2>', unsafe_allow_html=True)
    
    # Check if models have been trained
    if 'test_metrics' not in st.session_state:
        st.warning("⚠️ No models have been trained yet!")
        st.info("Please go to 'Model Predictions' page and train models first.")
    else:
        train_metrics = st.session_state['train_metrics']
        test_metrics = st.session_state['test_metrics']
        
        # Create comparison DataFrame from calculated metrics
        comparison_data = []
        for model_name in test_metrics.keys():
            comparison_data.append({
                'Model': model_name,
                'Train R²': train_metrics[model_name]['R²'],
                'Test R²': test_metrics[model_name]['R²'],
                'R² Gap': abs(train_metrics[model_name]['R²'] - test_metrics[model_name]['R²']),
                'Train RMSE': train_metrics[model_name]['RMSE'],
                'Test RMSE': test_metrics[model_name]['RMSE'],
                'Train MAE': train_metrics[model_name]['MAE'],
                'Test MAE': test_metrics[model_name]['MAE']
            })
        
        results_df = pd.DataFrame(comparison_data)
        results_df = results_df.sort_values('Test R²', ascending=False).reset_index(drop=True)
        
        # Top 5 models based on Test R²
        top_models = results_df.head(5)
        
        st.subheader("🏆 Top 5 Performing Models (Based on Test R²)")
        
        # Display top 5 models in columns with calculated metrics
        cols = st.columns(5)
        for idx, (_, model) in enumerate(top_models.iterrows()):
            with cols[idx]:
                st.markdown(f"### {model['Model']}")
                st.metric("Test R²", f"{model['Test R²']:.3f}")
                st.metric("Test RMSE", f"{model['Test RMSE']:.2f}")
                st.metric("R² Gap", f"{model['R² Gap']:.3f}")
        
        st.divider()
        
        st.subheader("📊 Detailed Performance Comparison")
        
        # Highlight top 5 models
        def highlight_top5(row):
            if row.name < 5:
                return ['background-color: #e8f5e8; font-weight: bold'] * len(row)
            return [''] * len(row)
        
        # Format the DataFrame for display
        display_df = results_df.copy()
        display_df['Train R²'] = display_df['Train R²'].apply(lambda x: f"{x:.4f}")
        display_df['Test R²'] = display_df['Test R²'].apply(lambda x: f"{x:.4f}")
        display_df['R² Gap'] = display_df['R² Gap'].apply(lambda x: f"{x:.4f}")
        display_df['Train RMSE'] = display_df['Train RMSE'].apply(lambda x: f"{x:.2f}")
        display_df['Test RMSE'] = display_df['Test RMSE'].apply(lambda x: f"{x:.2f}")
        display_df['Train MAE'] = display_df['Train MAE'].apply(lambda x: f"{x:.2f}")
        display_df['Test MAE'] = display_df['Test MAE'].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(
            display_df.style.apply(highlight_top5, axis=1),
            use_container_width=True
        )
        
        # Visual comparison of top 5 models
        st.subheader("📈 Performance Visualization (Top 5 Models)")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Test R² comparison
        axes[0,0].barh(top_models['Model'], top_models['Test R²'], color='#2E8B57')
        axes[0,0].set_xlabel('Test R² Score (Higher is Better)')
        axes[0,0].set_title('Test Set R² Comparison')
        axes[0,0].set_xlim([0, 1])
        axes[0,0].invert_yaxis()
        
        # R² Gap comparison (overfitting indicator)
        axes[0,1].barh(top_models['Model'], top_models['R² Gap'], color='#FF6B6B')
        axes[0,1].set_xlabel('R² Gap (Lower is Better)')
        axes[0,1].set_title('Overfitting Indicator (Train vs Test R² Gap)')
        axes[0,1].invert_yaxis()
        
        # Test RMSE comparison
        axes[1,0].barh(top_models['Model'], top_models['Test RMSE'], color='#3CB371')
        axes[1,0].set_xlabel('Test RMSE (Lower is Better)')
        axes[1,0].set_title('Test Set RMSE Comparison')
        axes[1,0].invert_yaxis()
        
        # Train vs Test R² comparison (side-by-side)
        x = np.arange(len(top_models['Model']))
        width = 0.35
        
        axes[1,1].bar(x - width/2, top_models['Train R²'], width, label='Train R²', color='#90EE90')
        axes[1,1].bar(x + width/2, top_models['Test R²'], width, label='Test R²', color='#2E8B57')
        axes[1,1].set_xlabel('Model')
        axes[1,1].set_ylabel('R² Score')
        axes[1,1].set_title('Train vs Test R² Comparison')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(top_models['Model'], rotation=45, ha='right')
        axes[1,1].legend()
        axes[1,1].set_ylim([0, 1])
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Best model recommendation
        best_model_row = results_df.iloc[0]
        st.subheader("🎯 Best Model Recommendation")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<div class="best-model">', unsafe_allow_html=True)
            st.markdown(f"### 🥇 {best_model_row['Model']}")
            st.metric("Test R²", f"{best_model_row['Test R²']:.4f}")
            st.metric("Test RMSE", f"{best_model_row['Test RMSE']:.2f}")
            st.metric("R² Gap", f"{best_model_row['R² Gap']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Why this model is recommended:")
            
            if best_model_row['R² Gap'] < 0.05:
                st.success("✅ **Excellent Generalization**: Minimal overfitting")
                st.write(f"The model shows only {best_model_row['R² Gap']:.3f} gap between train and test performance.")
            
            if best_model_row['Test R²'] > 0.85:
                st.success("✅ **High Predictive Power**: Explains most of the variance")
                st.write(f"R² of {best_model_row['Test R²']:.3f} indicates strong predictive ability.")
            
            if best_model_row['Test RMSE'] < 500:
                st.success("✅ **Low Error**: Accurate predictions")
                st.write(f"Average prediction error is only ${best_model_row['Test RMSE']:.2f}.")
            
            st.write("### Recommended for production because:")
            st.write("1. Best balance of accuracy and generalization")
            st.write("2. Minimal overfitting risk")
            st.write("3. Reliable performance on unseen data")
        
        # Export results option
        st.subheader("📥 Export Results")
        if st.button("💾 Download Performance Metrics as CSV"):
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Click to download",
                data=csv,
                file_name="model_performance_metrics.csv",
                mime="text/csv"
            )

elif page == "Hyperparameter Tuning":
    st.markdown('<h2 class="sub-header">Hyperparameter Tuning Configuration</h2>', unsafe_allow_html=True)
    
    st.info("""
    Hyperparameter tuning helps find the optimal parameters for each machine learning model 
    to achieve the best performance. This page shows the hyperparameter search spaces for each model.
    """)
    
    # Display hyperparameter grids
    param_grids = get_hyperparameter_grids()
    
    for model_name, params in param_grids.items():
        with st.expander(f"📊 {model_name} Hyperparameters"):
            st.write(f"**Number of parameter combinations to search:**")
            
            # Calculate total combinations
            total_combs = 1
            for key, values in params.items():
                total_combs *= len(values)
            
            st.write(f"Total possible combinations: **{total_combs:,}**")
            
            # Display parameters
            st.write("**Parameter search space:**")
            for key, values in params.items():
                st.write(f"- **{key}:** {values}")
            
            # Tuning recommendations
            st.write("**Tuning Recommendations:**")
            if model_name == 'Random Forest':
                st.write("- Focus on `n_estimators` and `max_depth` first")
                st.write("- Higher `n_estimators` usually improves performance but increases training time")
                st.write("- Use `max_features='sqrt'` for high-dimensional data")
            
            elif model_name == 'XGBoost':
                st.write("- `learning_rate` and `max_depth` are most important")
                st.write("- Lower `learning_rate` with higher `n_estimators` often works better")
                st.write("- `gamma` controls regularization (higher = more conservative)")
            
            elif model_name == 'Gradient Boosting':
                st.write("- Balance `learning_rate` and `n_estimators`")
                st.write("- Use `subsample` < 1.0 for stochastic gradient boosting")
                st.write("- `max_depth` typically between 3-5 works well")
            
            elif model_name == 'SVR':
                st.write("- `C` controls regularization (higher = less regularization)")
                st.write("- `epsilon` defines the margin of tolerance")
                st.write("- `kernel='rbf'` works well for non-linear problems")
            
            elif model_name == 'Decision Tree':
                st.write("- `max_depth` prevents overfitting")
                st.write("- `min_samples_split` and `min_samples_leaf` control tree growth")
                st.write("- `criterion='friedman_mse'` often works well for regression")
    
    # Performance metrics explanation
    st.subheader("📊 Performance Metrics Explained")
    
    metrics_cols = st.columns(3)
    
    with metrics_cols[0]:
        st.metric("R² Score", "0.0-1.0", "Higher is Better")
        st.write("""
        **Coefficient of Determination:**
        - Measures how well predictions approximate actual values
        - Range: 0 to 1 (higher is better)
        - 1 = Perfect prediction
        - 0 = No predictive power
        - Negative = Worse than average
        """)
    
    with metrics_cols[1]:
        st.metric("RMSE", "Same units as target", "Lower is Better")
        st.write("""
        **Root Mean Squared Error:**
        - Square root of average squared differences
        - Sensitive to outliers
        - In same units as target variable
        - Penalizes large errors more heavily
        """)
    
    with metrics_cols[2]:
        st.metric("MAE", "Same units as target", "Lower is Better")
        st.write("""
        **Mean Absolute Error:**
        - Average absolute differences
        - Less sensitive to outliers than RMSE
        - Easier to interpret
        - In same units as target variable
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Developed using Streamlit | BSD3523 Machine Learning Project</p>
    <p>Group: CSM1 | University Malaysia Pahang Al-Sultan Abdullah</p>
    <p style='font-size: 0.9em; color: #666;'>
        5 Models with Real-time Metrics: Random Forest, XGBoost, Gradient Boosting, SVR, Decision Tree
    </p>
</div>
""", unsafe_allow_html=True)
