# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
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
import warnings
warnings.filterwarnings('ignore')

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
    .tuned-model {
        background-color: #e8f5e9;
        padding: 5px;
        border-radius: 5px;
        margin-bottom: 5px;
    }
    .untuned-model {
        background-color: #fff3e0;
        padding: 5px;
        border-radius: 5px;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🌴 Palm Oil Price Prediction System</h1>', unsafe_allow_html=True)
st.markdown("""
This application predicts palm oil prices using machine learning models. 
Compare tuned vs untuned models to find the best performing one.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Data Overview", "Model Predictions", "Results Comparison"])

# Cache data loading
@st.cache_data
def load_data():
    """Load and prepare data"""
    try:
        # Try to load data
        df = pd.read_csv("price.csv")
        
        # Ensure Date column is datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        else:
            # If no Date column, create one with specified date range
            start_date = '2020-01-01'
            end_date = '2025-08-25'
            df['Date'] = pd.date_range(start=start_date, end=end_date, periods=len(df))
        
        # Add year and month columns
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        
        # Create a target column if not present
        if 'Price' not in df.columns and len(df.columns) > 0:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df['Price'] = df[numeric_cols[0]]
            else:
                # Create synthetic price data
                np.random.seed(42)
                base_price = 1000
                trend = np.linspace(0, 500, len(df))
                seasonal = 200 * np.sin(2 * np.pi * np.arange(len(df)) / 365)
                noise = np.random.normal(0, 50, len(df))
                df['Price'] = base_price + trend + seasonal + noise
        
        # Fill missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    except FileNotFoundError:
        st.error("File 'price.csv' not found. Creating sample data for demonstration.")
        
        # Create sample data with specified date range
        start_date = '2020-01-01'
        end_date = '2025-08-25'
        n_samples = 2000  # Approximately daily data
        
        dates = pd.date_range(start=start_date, end=end_date, periods=n_samples)
        
        np.random.seed(42)
        # Create realistic palm oil price pattern
        base_price = 1000
        trend = np.linspace(0, 500, n_samples)  # Upward trend over years
        seasonal = 200 * np.sin(2 * np.pi * np.arange(n_samples) / 365)  # Yearly seasonality
        monthly_seasonal = 100 * np.sin(2 * np.pi * np.arange(n_samples) / 30)  # Monthly fluctuations
        noise = np.random.normal(0, 50, n_samples)  # Random noise
        
        price = base_price + trend + seasonal + monthly_seasonal + noise
        
        df = pd.DataFrame({
            'Date': dates,
            'Solarradiation': np.random.uniform(100, 400, n_samples),
            'Solarenergy': np.random.uniform(5, 25, n_samples),
            'Uvindex': np.random.uniform(1, 12, n_samples),
            'Index Production': np.random.uniform(80, 120, n_samples),
            'Export Number (in Tonnes)': np.random.uniform(100000, 500000, n_samples),
            'USD': np.random.uniform(0.8, 1.2, n_samples),
            'Price': price
        })
        
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        
        return df

def get_hyperparameter_grids():
    """Define hyperparameter grids for tuning"""
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
def train_tuned_models(X_train, y_train, X_test, y_test, tuning_method='grid', cv_folds=3):
    """Train models WITH hyperparameter tuning"""
    models = {}
    best_params = {}
    test_metrics = {}
    training_times = {}
    
    param_grids = get_hyperparameter_grids()
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models_to_train = {
        'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'SVR': SVR(),
        'Decision Tree': DecisionTreeRegressor(random_state=42)
    }
    
    for model_name, model in models_to_train.items():
        # Progress indicator
        with st.spinner(f"Training {model_name} (tuned)..."):
            if tuning_method == 'grid':
                search = GridSearchCV(
                    model, 
                    param_grids[model_name], 
                    cv=cv_folds, 
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=0
                )
            else:
                search = RandomizedSearchCV(
                    model,
                    param_grids[model_name],
                    n_iter=10,
                    cv=cv_folds,
                    scoring='neg_mean_squared_error',
                    random_state=42,
                    n_jobs=-1,
                    verbose=0
                )
            
            start_time = time.time()
            search.fit(X_train_scaled, y_train)
            training_times[model_name] = time.time() - start_time
            
            models[f"{model_name} (Tuned)"] = search.best_estimator_
            best_params[model_name] = search.best_params_
            
            # Calculate TEST metrics only
            y_test_pred = search.best_estimator_.predict(X_test_scaled)
            
            test_metrics[f"{model_name} (Tuned)"] = {
                'R²': r2_score(y_test, y_test_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'MAE': mean_absolute_error(y_test, y_test_pred)
            }
    
    return models, best_params, test_metrics, scaler, training_times

@st.cache_resource
def train_untuned_models(X_train, y_train, X_test, y_test):
    """Train models WITHOUT hyperparameter tuning"""
    models = {}
    test_metrics = {}
    training_times = {}
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define default models
    default_models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),
        'SVR': SVR(kernel='rbf', C=10, epsilon=0.1),
        'Decision Tree': DecisionTreeRegressor(
            max_depth=5,
            min_samples_leaf=5,
            random_state=42
        )
    }
    
    for model_name, model in default_models.items():
        # Progress indicator
        with st.spinner(f"Training {model_name} (untuned)..."):
            start_time = time.time()
            model.fit(X_train_scaled, y_train)
            training_times[model_name] = time.time() - start_time
            
            models[f"{model_name} (Untuned)"] = model
            
            # Calculate TEST metrics only
            y_test_pred = model.predict(X_test_scaled)
            
            test_metrics[f"{model_name} (Untuned)"] = {
                'R²': r2_score(y_test, y_test_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'MAE': mean_absolute_error(y_test, y_test_pred)
            }
    
    return models, test_metrics, scaler, training_times

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
            buffer.append(f"**Date Range:** {df['Date'].min().date()} to {df['Date'].max().date()}\n\n")
        if 'Price' in df.columns:
            buffer.append(f"**Average Price:** $ {df['Price'].mean():.2f}\n\n")
            buffer.append(f"**Price Range:** ${df['Price'].min():.2f}-${df['Price'].max():.2f}")
        
        st.markdown("\n".join(buffer))
    
    # Line chart of price over time
    if 'Date' in df.columns and 'Price' in df.columns:
        st.subheader("📈 Price Trend Chart")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['Date'], df['Price'], linewidth=2, color='#2E8B57')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.set_title('Palm Oil Price Trend (2020-2025)')
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
                           'USD', 'Year', 'Month', 'Day']
        
        # Get only features that exist in dataframe
        available_features = [f for f in possible_features if f in df.columns]
        
        # Add any other numeric columns as features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Price' in numeric_cols:
            numeric_cols.remove('Price')
        available_features = list(set(available_features + numeric_cols))
        
        if len(available_features) < 1:
            st.error("Not enough features available for modeling.")
            st.info(f"Available columns: {df.columns.tolist()}")
        else:
            X = df[available_features]
            y = df['Price']
            
            # Handle any missing values
            X = X.fillna(X.median())
            y = y.fillna(y.median())
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Display data split information
            with st.expander("📊 Data Split Information"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Training Samples", len(X_train))
                with col2:
                    st.metric("Testing Samples", len(X_test))
                with col3:
                    st.metric("Total Features", len(available_features))
                
                st.write(f"**Features used:** {', '.join(available_features)}")
            
            # Training options
            st.sidebar.subheader("Training Options")
            train_tuned = st.sidebar.checkbox("Train Tuned Models", value=True)
            train_untuned = st.sidebar.checkbox("Train Untuned Models", value=True)
            tuning_method = st.sidebar.selectbox("Tuning Method", ["grid", "random"], index=0)
            cv_folds = st.sidebar.slider("CV Folds", min_value=3, max_value=10, value=3)
            
            # Initialize session state for models if not exists
            if 'all_models' not in st.session_state:
                st.session_state.all_models = {}
                st.session_state.all_test_metrics = {}
                st.session_state.all_scalers = {}
                st.session_state.model_trained = False
            
            # Train models button
            if st.button("🚀 Train Selected Models", type="primary"):
                if not train_tuned and not train_untuned:
                    st.warning("Please select at least one model type to train.")
                else:
                    # Clear previous results
                    st.session_state.all_models = {}
                    st.session_state.all_test_metrics = {}
                    st.session_state.all_scalers = {}
                    
                    # Train tuned models
                    if train_tuned:
                        with st.spinner("Training tuned models..."):
                            tuned_models, tuned_best_params, tuned_test_metrics, tuned_scaler, tuned_times = train_tuned_models(
                                X_train, y_train, X_test, y_test, tuning_method, cv_folds
                            )
                            st.session_state.all_models.update(tuned_models)
                            st.session_state.all_test_metrics.update(tuned_test_metrics)
                            st.session_state.all_scalers['tuned'] = tuned_scaler
                    
                    # Train untuned models
                    if train_untuned:
                        with st.spinner("Training untuned models..."):
                            untuned_models, untuned_test_metrics, untuned_scaler, untuned_times = train_untuned_models(
                                X_train, y_train, X_test, y_test
                            )
                            st.session_state.all_models.update(untuned_models)
                            st.session_state.all_test_metrics.update(untuned_test_metrics)
                            st.session_state.all_scalers['untuned'] = untuned_scaler
                    
                    st.session_state.model_trained = True
                    st.success("✅ All selected models trained successfully!")
            
            # Display results if models are trained
            if st.session_state.model_trained and st.session_state.all_models:
                st.subheader("📊 Test Set Performance")
                
                # Create a DataFrame for display
                performance_data = []
                for model_name, metrics in st.session_state.all_test_metrics.items():
                    performance_data.append({
                        'Model': model_name,
                        'R²': metrics['R²'],
                        'RMSE': metrics['RMSE'],
                        'MAE': metrics['MAE']
                    })
                
                performance_df = pd.DataFrame(performance_data)
                performance_df = performance_df.sort_values('R²', ascending=False)
                
                # Display performance metrics
                st.dataframe(
                    performance_df.style.format({
                        'R²': '{:.4f}',
                        'RMSE': '{:.2f}',
                        'MAE': '{:.2f}'
                    }),
                    use_container_width=True
                )
                
                # Select a model for detailed predictions
                st.subheader("🔍 Detailed Model Analysis")
                
                selected_model_name = st.selectbox(
                    "Choose a model to view detailed predictions:",
                    list(st.session_state.all_models.keys())
                )
                
                # Get the selected model and its scaler
                selected_model = st.session_state.all_models[selected_model_name]
                
                # Determine which scaler to use based on model type
                if "(Tuned)" in selected_model_name:
                    scaler = st.session_state.all_scalers.get('tuned')
                else:
                    scaler = st.session_state.all_scalers.get('untuned')
                
                if scaler is not None:
                    # Make predictions
                    X_test_scaled = scaler.transform(X_test)
                    y_pred = selected_model.predict(X_test_scaled)
                    
                    # Get test metrics for the selected model
                    test_metrics = st.session_state.all_test_metrics[selected_model_name]
                    
                    # Display metrics in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("### R² Score")
                        st.metric("Test R²", f"{test_metrics['R²']:.4f}")
                    
                    with col2:
                        st.markdown("### RMSE")
                        st.metric("Test RMSE", f"{test_metrics['RMSE']:.2f}")
                    
                    with col3:
                        st.markdown("### MAE")
                        st.metric("Test MAE", f"{test_metrics['MAE']:.2f}")
                    
                    # Plot predictions vs actual
                    st.subheader("📈 Prediction Visualization")
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                    
                    # Scatter plot
                    ax1.scatter(y_test, y_pred, alpha=0.5, color='#2E8B57', s=30)
                    ax1.plot([y_test.min(), y_test.max()], 
                            [y_test.min(), y_test.max()], 
                            'r--', lw=2, label='Perfect Prediction')
                    ax1.set_xlabel('Actual Price ($)')
                    ax1.set_ylabel('Predicted Price ($)')
                    ax1.set_title(f'{selected_model_name}: Test Set Predictions')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Residual plot
                    residuals = y_test - y_pred
                    ax2.scatter(y_pred, residuals, alpha=0.5, color='#FF6B6B', s=30)
                    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
                    ax2.set_xlabel('Predicted Price ($)')
                    ax2.set_ylabel('Residuals')
                    ax2.set_title(f'{selected_model_name}: Residual Plot')
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Show sample predictions
                    st.subheader("📋 Sample Test Predictions")
                    
                    # Select number of samples to show
                    n_samples = st.slider("Number of samples to show:", 10, 50, 20)
                    
                    results_df = pd.DataFrame({
                        'Actual Price': y_test.values[:n_samples],
                        'Predicted Price': y_pred[:n_samples],
                        'Difference': y_pred[:n_samples] - y_test.values[:n_samples],
                        'Error %': abs((y_pred[:n_samples] - y_test.values[:n_samples]) / y_test.values[:n_samples] * 100)
                    })
                    
                    st.dataframe(results_df.style.format({
                        'Actual Price': '${:.2f}',
                        'Predicted Price': '${:.2f}',
                        'Difference': '${:.2f}',
                        'Error %': '{:.2f}%'
                    }), use_container_width=True)
                    
                    # Feature importance for tree-based models
                    if hasattr(selected_model, 'feature_importances_'):
                        st.subheader("🎯 Feature Importance")
                        importance = pd.DataFrame({
                            'Feature': available_features,
                            'Importance': selected_model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.barh(importance['Feature'][:10], importance['Importance'][:10], color='#2E8B57')
                        ax.set_xlabel('Importance Score')
                        ax.set_title(f'{selected_model_name}: Top 10 Feature Importance')
                        ax.invert_yaxis()
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        st.dataframe(importance.head(10).style.format({'Importance': '{:.4f}'}), 
                                   use_container_width=True)
                    
                    # Interactive prediction
                    st.subheader("🎮 Make a Custom Prediction")
                    
                    # Create input values for features
                    col1, col2 = st.columns(2)
                    input_values = {}
                    
                    with col1:
                        for i, feature in enumerate(available_features[:len(available_features)//2]):
                            if feature in X_train.columns:
                                min_val = float(X_train[feature].min())
                                max_val = float(X_train[feature].max())
                                mean_val = float(X_train[feature].mean())
                                input_values[feature] = st.slider(
                                    feature, min_val, max_val, mean_val, (max_val-min_val)/100
                                )
                    
                    with col2:
                        for i, feature in enumerate(available_features[len(available_features)//2:]):
                            if feature in X_train.columns:
                                min_val = float(X_train[feature].min())
                                max_val = float(X_train[feature].max())
                                mean_val = float(X_train[feature].mean())
                                input_values[feature] = st.slider(
                                    feature, min_val, max_val, mean_val, (max_val-min_val)/100
                                )
                    
                    if st.button("🔮 Predict Price", type="secondary"):
                        # Create input array
                        input_array = np.array([[input_values[feat] for feat in available_features]])
                        
                        # Scale input
                        input_scaled = scaler.transform(input_array)
                        
                        # Make prediction
                        prediction = selected_model.predict(input_scaled)[0]
                        
                        # Display result
                        st.success(f"**Predicted Price:** ${prediction:.2f}")
                        
                        # Show comparison with average price
                        avg_price = y.mean()
                        st.info(f"Average historical price: ${avg_price:.2f}")
                        diff = prediction - avg_price
                        if diff > 0:
                            st.write(f"Prediction is ${diff:.2f} higher than average")
                        else:
                            st.write(f"Prediction is ${-diff:.2f} lower than average")
                else:
                    st.error("Scaler not found for the selected model.")
            elif st.session_state.model_trained and not st.session_state.all_models:
                st.info("No models have been trained yet. Please select model types and click 'Train Selected Models'.")
            else:
                st.info("Please select model types and click '🚀 Train Selected Models' button to start training.")

elif page == "Results Comparison":
    st.markdown('<h2 class="sub-header">Tuned vs Untuned Model Comparison</h2>', unsafe_allow_html=True)
    
    # Check if models have been trained
    if 'all_test_metrics' not in st.session_state or not st.session_state.all_test_metrics:
        st.warning("⚠️ No models have been trained yet!")
        st.info("Please go to 'Model Predictions' page, select models, and train them first.")
    else:
        all_test_metrics = st.session_state.all_test_metrics
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in all_test_metrics.items():
            comparison_data.append({
                'Model': model_name,
                'Type': 'Tuned' if '(Tuned)' in model_name else 'Untuned',
                'R²': metrics['R²'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE']
            })
        
        results_df = pd.DataFrame(comparison_data)
        
        # Sort by R² score
        results_df = results_df.sort_values('R²', ascending=False).reset_index(drop=True)
        
        # Display comparison table
        st.subheader("📊 All Models Performance Comparison")
        
        # Format the DataFrame
        display_df = results_df.copy()
        display_df['R²'] = display_df['R²'].apply(lambda x: f"{x:.4f}")
        display_df['RMSE'] = display_df['RMSE'].apply(lambda x: f"{x:.2f}")
        display_df['MAE'] = display_df['MAE'].apply(lambda x: f"{x:.2f}")
        
        # Apply styling based on model type
        def color_model_type(val):
            if val == 'Tuned':
                return 'background-color: #e8f5e9'
            else:
                return 'background-color: #fff3e0'
        
        styled_df = display_df.style.applymap(color_model_type, subset=['Type'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Separate tuned and untuned models for comparison
        tuned_df = results_df[results_df['Type'] == 'Tuned'].copy()
        untuned_df = results_df[results_df['Type'] == 'Untuned'].copy()
        
        # Prepare data for visualization
        st.subheader("📈 Tuned vs Untuned Performance Comparison")
        
        # Create base model names (without tuned/untuned suffix)
        base_models = []
        for model_name in results_df['Model']:
            base_name = model_name.replace(' (Tuned)', '').replace(' (Untuned)', '')
            if base_name not in base_models:
                base_models.append(base_name)
        
        # Create visualization
        if tuned_df.shape[0] > 0 and untuned_df.shape[0] > 0:
            # Create comparison charts
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Bar chart comparing R² scores
            model_names = []
            tuned_r2 = []
            untuned_r2 = []
            
            for base_model in base_models:
                tuned_model = f"{base_model} (Tuned)"
                untuned_model = f"{base_model} (Untuned)"
                
                if tuned_model in all_test_metrics and untuned_model in all_test_metrics:
                    model_names.append(base_model)
                    tuned_r2.append(all_test_metrics[tuned_model]['R²'])
                    untuned_r2.append(all_test_metrics[untuned_model]['R²'])
            
            if model_names:
                x = np.arange(len(model_names))
                width = 0.35
                
                axes[0, 0].bar(x - width/2, tuned_r2, width, label='Tuned', color='#4CAF50')
                axes[0, 0].bar(x + width/2, untuned_r2, width, label='Untuned', color='#FF9800')
                axes[0, 0].set_xlabel('Model')
                axes[0, 0].set_ylabel('R² Score')
                axes[0, 0].set_title('R² Comparison: Tuned vs Untuned')
                axes[0, 0].set_xticks(x)
                axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
                axes[0, 0].legend()
                axes[0, 0].set_ylim([0, 1])
                axes[0, 0].grid(True, alpha=0.3)
                
                # RMSE comparison
                tuned_rmse = []
                untuned_rmse = []
                
                for base_model in base_models:
                    tuned_model = f"{base_model} (Tuned)"
                    untuned_model = f"{base_model} (Untuned)"
                    
                    if tuned_model in all_test_metrics and untuned_model in all_test_metrics:
                        tuned_rmse.append(all_test_metrics[tuned_model]['RMSE'])
                        untuned_rmse.append(all_test_metrics[untuned_model]['RMSE'])
                
                axes[0, 1].bar(x - width/2, tuned_rmse, width, label='Tuned', color='#4CAF50')
                axes[0, 1].bar(x + width/2, untuned_rmse, width, label='Untuned', color='#FF9800')
                axes[0, 1].set_xlabel('Model')
                axes[0, 1].set_ylabel('RMSE')
                axes[0, 1].set_title('RMSE Comparison: Tuned vs Untuned')
                axes[0, 1].set_xticks(x)
                axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                # Best overall models
                top_n = min(5, len(results_df))
                top_models = results_df.head(top_n)
                
                # Top models by R²
                axes[1, 0].barh(range(top_n), top_models['R²'][::-1], color='#2E8B57')
                axes[1, 0].set_yticks(range(top_n))
                axes[1, 0].set_yticklabels(top_models['Model'][::-1])
                axes[1, 0].set_xlabel('R² Score')
                axes[1, 0].set_title(f'Top {top_n} Models (by R²)')
                axes[1, 0].set_xlim([0, 1])
                axes[1, 0].grid(True, alpha=0.3)
                
                # Type distribution in top models
                type_counts = top_models['Type'].value_counts()
                axes[1, 1].pie(type_counts.values, labels=type_counts.index, 
                              colors=['#4CAF50', '#FF9800'], autopct='%1.1f%%')
                axes[1, 1].set_title(f'Type Distribution in Top {top_n} Models')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Performance improvement analysis
            st.subheader("📈 Tuning Effectiveness Analysis")
            
            improvement_data = []
            for base_model in base_models:
                tuned_model = f"{base_model} (Tuned)"
                untuned_model = f"{base_model} (Untuned)"
                
                if tuned_model in all_test_metrics and untuned_model in all_test_metrics:
                    r2_improvement = (all_test_metrics[tuned_model]['R²'] - 
                                     all_test_metrics[untuned_model]['R²'])
                    rmse_improvement = (all_test_metrics[untuned_model]['RMSE'] - 
                                       all_test_metrics[tuned_model]['RMSE'])
                    
                    improvement_data.append({
                        'Model': base_model,
                        'R² Improvement': r2_improvement,
                        'RMSE Improvement': rmse_improvement,
                        'Tuned R²': all_test_metrics[tuned_model]['R²'],
                        'Untuned R²': all_test_metrics[untuned_model]['R²']
                    })
            
            if improvement_data:
                improvement_df = pd.DataFrame(improvement_data)
                improvement_df = improvement_df.sort_values('R² Improvement', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**R² Improvement from Tuning:**")
                    for _, row in improvement_df.iterrows():
                        if row['R² Improvement'] > 0:
                            st.success(f"✅ {row['Model']}: +{row['R² Improvement']:.4f}")
                        else:
                            st.warning(f"⚠️ {row['Model']}: {row['R² Improvement']:.4f}")
                
                with col2:
                    st.write("**RMSE Improvement from Tuning:**")
                    for _, row in improvement_df.iterrows():
                        if row['RMSE Improvement'] > 0:
                            st.success(f"✅ {row['Model']}: -{row['RMSE Improvement']:.2f}")
                        else:
                            st.warning(f"⚠️ {row['Model']}: +{abs(row['RMSE Improvement']):.2f}")
                
                # Best model recommendation
                st.subheader("🎯 Best Model Recommendation")
                
                best_model_row = results_df.iloc[0]
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown('<div class="best-model">', unsafe_allow_html=True)
                    st.markdown(f"### 🥇 {best_model_row['Model']}")
                    st.metric("R²", f"{best_model_row['R²']:.4f}")
                    st.metric("RMSE", f"{best_model_row['RMSE']:.2f}")
                    st.metric("MAE", f"{best_model_row['MAE']:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### Why this model is recommended:")
                    
                    if best_model_row['R²'] > 0.85:
                        st.success("✅ **High Predictive Power**: Explains most of the variance")
                    
                    if best_model_row['RMSE'] < 100:
                        st.success("✅ **Low Error**: Highly accurate predictions")
                    elif best_model_row['RMSE'] < 200:
                        st.success("✅ **Good Accuracy**: Reasonable prediction error")
                    
                    if "(Tuned)" in best_model_row['Model']:
                        st.info("🔧 **Tuned Model**: Hyperparameters optimized for best performance")
                        # Find the untuned version for comparison
                        base_name = best_model_row['Model'].replace(' (Tuned)', '')
                        untuned_name = f"{base_name} (Untuned)"
                        if untuned_name in all_test_metrics:
                            untuned_r2 = all_test_metrics[untuned_name]['R²']
                            improvement = best_model_row['R²'] - untuned_r2
                            st.write(f"Tuning improved R² by {improvement:.4f} over untuned version")
                    
                    st.write("### Key Strengths:")
                    st.write("1. Best overall performance on test data")
                    st.write("2. Balanced error metrics")
                    st.write("3. Reliable predictions")
        
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Developed using Streamlit | BSD3523 Machine Learning Project</p>
    <p>Group: CSM1 | University Malaysia Pahang Al-Sultan Abdullah</p>
    <p style='font-size: 0.9em; color: #666;'>
        Comparing Tuned vs Untuned Models: Random Forest, XGBoost, Gradient Boosting, SVR, Decision Tree
    </p>
</div>
""", unsafe_allow_html=True)
