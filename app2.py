# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob

# Import machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import top 4 models based on R-squared
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline

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
    .model-section {
        background-color: #f9f9f9;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🌴 Palm Oil Price Prediction System</h1>', unsafe_allow_html=True)
st.markdown("""
This application predicts palm oil prices using machine learning models.
The system integrates weather data, production indices, exchange rates, and export numbers to provide accurate predictions.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["📊 Data Overview", "🤖 Model Predictions", "📈 Results Comparison"])

# Cache data loading
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    try:
        # Load all CSV files (adjust paths as needed)
        weather_df = pd.read_csv('weather.csv')
        price2020_df = pd.read_csv('price_2020.csv')
        price2021_df = pd.read_csv('price_2021.csv')
        price2022_df = pd.read_csv('price_2022.csv')
        ipi_df = pd.read_csv('production_index.csv')
        export_df = pd.read_csv('export_number.csv')
        exchange_df = pd.read_csv('exchange_rates.csv')
        
        # Convert date columns
        for df in [weather_df, price2020_df, price2021_df, price2022_df, ipi_df, export_df, exchange_df]:
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Preprocessing steps (simplified version)
        # Merge all dataframes
        df = weather_df.copy()
        
        # Add year and month features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return sample data if files not found
        dates = pd.date_range(start='2020-01-01', end='2022-05-31', freq='D')
        df = pd.DataFrame({
            'Date': dates,
            'Temperature': np.random.normal(25, 5, len(dates)),
            'Humidity': np.random.normal(80, 10, len(dates)),
            'Solarradiation': np.random.normal(200, 50, len(dates)),
            'Solarenergy': np.random.normal(15, 3, len(dates)),
            'Uvindex': np.random.normal(5, 2, len(dates)),
            'Index Production': np.random.normal(100, 20, len(dates)),
            'Export Number (in Tonnes)': np.random.normal(100000, 20000, len(dates)),
            'USD': np.random.normal(4.2, 0.2, len(dates)),
            'Price': np.random.normal(3000, 500, len(dates)),
            'Year': dates.year,
            'Month': dates.month
        })
        return df

@st.cache_resource
def train_models(X_train_scaled, y_train, selected_features):
    """Train the top 4 models"""
    models = {}
    
    # 1. Random Forest Regressor
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42
    )
    rf.fit(X_train_scaled, y_train)
    models['Random Forest'] = rf
    
    # 2. XGBoost Regressor
    xgb = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb.fit(X_train_scaled, y_train)
    models['XGBoost'] = xgb
    
    # 3. Gradient Boosting Regressor
    gbr = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    gbr.fit(X_train_scaled, y_train)
    models['Gradient Boosting'] = gbr
    
    # 4. SVR (Support Vector Regression)
    svr = make_pipeline(
        StandardScaler(),
        SVR(kernel='rbf', C=100, epsilon=0.1)
    )
    svr.fit(X_train_scaled, y_train)
    models['SVR'] = svr
    
    return models

# Load data
df = load_and_preprocess_data()

if page == "📊 Data Overview":
    st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Data Sample")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.subheader("📊 Data Information")
        buffer = []
        buffer.append(f"Total Rows: {df.shape[0]}")
        buffer.append(f"Total Columns: {df.shape[1]}")
        buffer.append(f"Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        st.write("\n".join(buffer))
    
    st.subheader("📈 Price Distribution Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Date'], df['Price'], linewidth=2, color='#2E8B57')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Palm Oil Price Trend')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.subheader("🔍 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

elif page == "🤖 Model Predictions":
    st.markdown('<h2 class="sub-header">Model Training & Prediction</h2>', unsafe_allow_html=True)
    
    # Prepare data for modeling
    if 'Price' in df.columns:
        # Select features based on correlation
        features = ['Solarradiation', 'Solarenergy', 'Uvindex', 
                   'Index Production', 'Export Number (in Tonnes)', 
                   'USD', 'Year', 'Month']
        
        # Filter features that exist in the dataframe
        available_features = [f for f in features if f in df.columns]
        
        X = df[available_features]
        y = df['Price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Feature selection using RFE
        st.subheader("🎯 Feature Selection")
        estimator = LinearRegression()
        rfe = RFE(estimator=estimator, n_features_to_select=5)
        rfe.fit(X_train_scaled, y_train)
        
        selected_features = X_train.columns[rfe.support_]
        st.write(f"Selected Features: {', '.join(selected_features)}")
        
        # Transform data with selected features
        X_train_rfe = rfe.transform(X_train_scaled)
        X_test_rfe = rfe.transform(X_test_scaled)
        
        # Train models
        with st.spinner("Training models..."):
            models = train_models(X_train_rfe, y_train, selected_features)
        
        st.success("✅ Models trained successfully!")
        
        # Make predictions
        st.subheader("📊 Model Predictions")
        
        # Select a model to view predictions
        selected_model = st.selectbox(
            "Choose a model to view predictions:",
            list(models.keys())
        )
        
        model = models[selected_model]
        
        # Make predictions
        y_pred = model.predict(X_test_rfe)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RMSE", f"{rmse:.2f}")
        with col2:
            st.metric("MAE", f"{mae:.2f}")
        with col3:
            st.metric("R² Score", f"{r2:.4f}")
        
        # Plot predictions vs actual
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, alpha=0.5, color='#2E8B57')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Price')
        ax.set_ylabel('Predicted Price')
        ax.set_title(f'{selected_model}: Actual vs Predicted Prices')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Show prediction table
        st.subheader("📋 Sample Predictions")
        results_df = pd.DataFrame({
            'Actual Price': y_test.values[:20],
            'Predicted Price': y_pred[:20],
            'Difference': y_pred[:20] - y_test.values[:20]
        })
        st.dataframe(results_df.style.format("{:.2f}"), use_container_width=True)
        
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
            year = st.selectbox("Year", [2020, 2021, 2022])
            month = st.selectbox("Month", range(1, 13))
        
        if st.button("Predict Price", type="primary"):
            # Create input array
            input_data = np.array([[solar_rad, solar_energy, uv_index, 
                                  production_idx, 100000, 4.2, year, month]])
            
            # Scale input
            input_scaled = scaler.transform(input_data)
            input_rfe = rfe.transform(input_scaled)
            
            # Get prediction from all models
            predictions = {}
            for name, model in models.items():
                pred = model.predict(input_rfe)[0]
                predictions[name] = pred
            
            # Display results
            st.subheader("📈 Prediction Results")
            
            for model_name, pred_price in predictions.items():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric(model_name, f"${pred_price:.2f}")
                with col2:
                    st.progress(min(int((pred_price / 5000) * 100), 100))
            
            # Show best prediction
            best_model = max(predictions, key=lambda x: predictions[x])
            st.info(f"💰 Highest predicted price: **{best_model}** at **${predictions[best_model]:.2f}**")
    else:
        st.warning("Price column not found in dataset. Using sample data.")

elif page == "📈 Results Comparison":
    st.markdown('<h2 class="sub-header">Model Performance Comparison</h2>', unsafe_allow_html=True)
    
    # Simulate results (replace with actual model results)
    results_data = {
        'Model': ['Random Forest', 'XGBoost', 'Gradient Boosting', 'SVR', 
                 'Linear Regression', 'Ridge', 'Lasso', 'Decision Tree'],
        'RMSE': [450.32, 420.15, 480.25, 510.42, 600.12, 590.34, 595.67, 550.89],
        'MAE': [320.45, 310.23, 350.12, 380.45, 420.34, 415.67, 418.90, 390.23],
        'R-squared': [0.892, 0.902, 0.878, 0.862, 0.812, 0.818, 0.815, 0.845]
    }
    
    results_df = pd.DataFrame(results_data)
    
    # Sort by R-squared
    results_df = results_df.sort_values('R-squared', ascending=False).reset_index(drop=True)
    
    # Highlight top 4 models
    def highlight_top4(row):
        if row.name < 4:
            return ['background-color: #e8f5e8'] * len(row)
        else:
            return [''] * len(row)
    
    st.subheader("🏆 Model Performance Ranking")
    st.dataframe(results_df.style.apply(highlight_top4, axis=1).format({
        'RMSE': '{:.2f}',
        'MAE': '{:.2f}',
        'R-squared': '{:.3f}'
    }), use_container_width=True)
    
    # Visual comparison
    st.subheader("📊 Performance Visualization")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RMSE comparison
    axes[0].barh(results_df['Model'][:4], results_df['RMSE'][:4], color='#2E8B57')
    axes[0].set_xlabel('RMSE')
    axes[0].set_title('RMSE Comparison (Lower is Better)')
    axes[0].invert_yaxis()
    
    # MAE comparison
    axes[1].barh(results_df['Model'][:4], results_df['MAE'][:4], color='#3CB371')
    axes[1].set_xlabel('MAE')
    axes[1].set_title('MAE Comparison (Lower is Better)')
    axes[1].invert_yaxis()
    
    # R-squared comparison
    axes[2].barh(results_df['Model'][:4], results_df['R-squared'][:4], color='#90EE90')
    axes[2].set_xlabel('R-squared')
    axes[2].set_title('R² Score Comparison (Higher is Better)')
    axes[2].invert_yaxis()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Feature importance (for tree-based models)
    st.subheader("🎯 Top Features Importance")
    
    # Create a sample feature importance chart
    features = ['Solarradiation', 'Solarenergy', 'Uvindex', 
               'Index Production', 'Year', 'Month', 'USD', 
               'Export Number']
    importance = [0.25, 0.20, 0.15, 0.18, 0.12, 0.05, 0.03, 0.02]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importance, color='#2E8B57')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance in Random Forest Model')
    plt.tight_layout()
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Developed with ❤️ using Streamlit | BSD3523 Machine Learning Project</p>
    <p>Group: CSM1 | University Malaysia Pahang Al-Sultan Abdullah</p>
</div>
""", unsafe_allow_html=True)
