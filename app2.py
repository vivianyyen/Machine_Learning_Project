# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Set page configuration
st.set_page_config(
    page_title="ML Manufacturing Project",
    page_icon="🏭",
    layout="wide"
)

# Title and description
st.title("🏭 BSD3523 MACHINE LEARNING GROUP PROJECT")
st.markdown("**Group Name:** CSM1")
st.markdown("""
**Group Members:**
- YIP YOONG ENG (SD23048) - Group Leader
- MUHAMMAS AMIRUL AMIER BIN MOHD HUSNI (SD23011)
- ALIYA AFIFAH BINTI AL ABAS (SD23062)
- NUR IZZATI BINTI ZAKARIA (SD23007)
- ALIA AYUNNI BINTI MOHD SHUKRI (SD23054)
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to:",
    ["Introduction", "Data Integration", "Data Preprocessing", "EDA", 
     "Feature Engineering", "Model Training", "Results", "Predictions"]
)

# Initialize session state for data
if 'df' not in st.session_state:
    st.session_state.df = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'results' not in st.session_state:
    st.session_state.results = {}

# Helper functions
def load_and_preprocess_data():
    """Simulate data loading and preprocessing"""
    # Note: In actual deployment, you would load real CSV files
    # For this template, we'll create synthetic data
    st.warning("⚠️ **Note:** In a real deployment, upload actual CSV files")
    
    # Create synthetic data for demonstration
    dates = pd.date_range(start='2020-01-01', end='2022-05-31', freq='D')
    n_days = len(dates)
    
    synthetic_data = {
        'Date': dates,
        'Temperature': np.random.normal(25, 5, n_days),
        'Humidity': np.random.normal(70, 10, n_days),
        'Dew': np.random.normal(20, 3, n_days),
        'Sealevelpressure': np.random.normal(1013, 10, n_days),
        'Solarradiation': np.random.normal(200, 50, n_days),
        'Solarenergy': np.random.normal(15, 4, n_days),
        'Uvindex': np.random.normal(5, 2, n_days),
        'Cloudcover': np.random.normal(50, 20, n_days),
        'Moonphase': np.random.uniform(0, 1, n_days),
        'Precipitation': np.random.exponential(2, n_days),
        'Windspeed': np.random.normal(10, 3, n_days),
        'Winddirection': np.random.uniform(0, 360, n_days),
        'Index Production': np.random.normal(100, 20, n_days),
        'Export Number (in Tonnes)': np.random.normal(50000, 10000, n_days),
        'USD': np.random.normal(4.2, 0.1, n_days),
    }
    
    # Add price with some seasonality and trend
    trend = np.linspace(100, 150, n_days)
    seasonality = 20 * np.sin(2 * np.pi * np.arange(n_days) / 365)
    noise = np.random.normal(0, 5, n_days)
    synthetic_data['Price'] = trend + seasonality + noise
    
    df = pd.DataFrame(synthetic_data)
    
    # Add some missing values for demonstration
    for col in ['Index Production', 'Export Number (in Tonnes)', 'USD', 'Sealevelpressure']:
        mask = np.random.choice([True, False], size=n_days, p=[0.05, 0.95])
        df.loc[mask, col] = np.nan
    
    st.session_state.df = df
    return df

# Section 1: Introduction
if section == "Introduction":
    st.header("📊 Project Overview")
    st.markdown("""
    This project focuses on **predicting palm oil prices** using machine learning techniques.
    
    ### Dataset Description
    The dataset combines multiple sources:
    1. **Weather Data** - Daily weather measurements
    2. **Price Data** - Daily palm oil prices (2020-2022)
    3. **Economic Indicators** - Exchange rates, production index, export numbers
    
    ### Objectives
    - Integrate multiple data sources
    - Perform exploratory data analysis
    - Build and compare multiple ML models
    - Predict palm oil prices based on various factors
    """)
    
    # Load data button
    if st.button("📁 Load Sample Data (Synthetic)"):
        with st.spinner("Loading and preprocessing data..."):
            df = load_and_preprocess_data()
            st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
            st.dataframe(df.head(), use_container_width=True)

# Section 2: Data Integration
elif section == "Data Integration":
    st.header("🔄 Data Integration")
    
    if st.session_state.df is None:
        st.info("Please load data from the Introduction section first.")
    else:
        df = st.session_state.df
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Overview")
            st.write(f"**Shape:** {df.shape}")
            st.write(f"**Date Range:** {df['Date'].min().date()} to {df['Date'].max().date()}")
            
            # Show data types
            st.subheader("Data Types")
            dtype_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
            st.dataframe(dtype_df, use_container_width=True)
        
        with col2:
            st.subheader("Column Information")
            columns = st.multiselect(
                "Select columns to view:",
                df.columns.tolist(),
                default=['Date', 'Price', 'Temperature', 'Index Production']
            )
            if columns:
                st.dataframe(df[columns].head(10), use_container_width=True)
        
        # Missing values
        st.subheader("Missing Values")
        missing_df = pd.DataFrame(df.isnull().sum(), columns=['Missing Values'])
        missing_df['Percentage'] = (missing_df['Missing Values'] / len(df)) * 100
        st.dataframe(missing_df, use_container_width=True)

# Section 3: Data Preprocessing
elif section == "Data Preprocessing":
    st.header("🧹 Data Preprocessing")
    
    if st.session_state.df is None:
        st.info("Please load data from the Introduction section first.")
    else:
        df = st.session_state.df.copy()
        
        st.subheader("Current Data")
        st.dataframe(df.head(), use_container_width=True)
        
        # Data type conversion
        st.subheader("Data Type Conversion")
        if st.checkbox("Convert Export Number to numeric"):
            df['Export Number (in Tonnes)'] = pd.to_numeric(
                df['Export Number (in Tonnes)'], errors='coerce'
            )
            st.success("Export Number converted to numeric")
        
        # Missing value imputation
        st.subheader("Missing Value Imputation")
        impute_method = st.selectbox(
            "Select imputation method:",
            ["Median", "Mean", "Forward Fill", "Drop"]
        )
        
        if st.button("Apply Imputation"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if impute_method == "Median":
                for col in numeric_cols:
                    df[col] = df[col].fillna(df[col].median())
            elif impute_method == "Mean":
                for col in numeric_cols:
                    df[col] = df[col].fillna(df[col].mean())
            elif impute_method == "Forward Fill":
                df = df.fillna(method='ffill')
            else:  # Drop
                df = df.dropna()
            
            st.success(f"Imputation applied using {impute_method} method")
            st.dataframe(df.isnull().sum().to_frame('Missing Values'), use_container_width=True)
            st.session_state.df = df

# Section 4: EDA
elif section == "EDA":
    st.header("📈 Exploratory Data Analysis")
    
    if st.session_state.df is None:
        st.info("Please load data from the Introduction section first.")
    else:
        df = st.session_state.df
        
        # Distribution plots
        st.subheader("Distribution of Numerical Variables")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        selected_col = st.selectbox("Select variable to plot:", numeric_cols)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        ax1.hist(df[selected_col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        ax1.set_xlabel(selected_col)
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Distribution of {selected_col}')
        
        # Box plot
        ax2.boxplot(df[selected_col].dropna())
        ax2.set_ylabel(selected_col)
        ax2.set_title(f'Box Plot of {selected_col}')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Correlation matrix
        st.subheader("Correlation Matrix")
        if st.checkbox("Show correlation matrix"):
            corr_matrix = df.select_dtypes(include=[np.number]).corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                       square=True, ax=ax)
            plt.title("Correlation Matrix")
            st.pyplot(fig)
        
        # Time series plot
        st.subheader("Time Series Analysis")
        if 'Date' in df.columns and 'Price' in df.columns:
            time_col = st.selectbox("Select time variable:", ['Price', 'Temperature', 'Index Production'])
            
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(df['Date'], df[time_col])
            ax.set_xlabel('Date')
            ax.set_ylabel(time_col)
            ax.set_title(f'{time_col} Over Time')
            plt.xticks(rotation=45)
            st.pyplot(fig)

# Section 5: Feature Engineering
elif section == "Feature Engineering":
    st.header("⚙️ Feature Engineering")
    
    if st.session_state.df is None:
        st.info("Please load data from the Introduction section first.")
    else:
        df = st.session_state.df.copy()
        
        st.subheader("Create Time-based Features")
        
        if st.checkbox("Extract date features"):
            if 'Date' in df.columns:
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
                df['Day'] = df['Date'].dt.day
                df['DayOfWeek'] = df['Date'].dt.dayofweek
                
                st.success("Date features created: Year, Month, Day, DayOfWeek")
                st.dataframe(df[['Date', 'Year', 'Month', 'Day', 'DayOfWeek']].head(), 
                           use_container_width=True)
        
        st.subheader("Feature Selection")
        
        # Prepare data for feature selection
        if 'Price' in df.columns:
            # Drop non-numeric columns for correlation analysis
            df_numeric = df.select_dtypes(include=[np.number])
            
            if len(df_numeric.columns) > 1:
                # Calculate correlation with target
                correlations = df_numeric.corr()['Price'].drop('Price', errors='ignore')
                
                st.write("**Correlation with Price:**")
                corr_df = pd.DataFrame(correlations.sort_values(ascending=False), 
                                     columns=['Correlation'])
                st.dataframe(corr_df, use_container_width=True)
                
                # Select features based on threshold
                threshold = st.slider("Correlation threshold:", 0.0, 1.0, 0.1)
                selected_features = correlations[abs(correlations) >= threshold].index.tolist()
                
                st.write(f"**Selected features (|correlation| ≥ {threshold}):**")
                st.write(selected_features)
                
                # Store selected features in session state
                st.session_state.selected_features = selected_features

# Section 6: Model Training
elif section == "Model Training":
    st.header("🤖 Model Training")
    
    if st.session_state.df is None:
        st.info("Please load data from the Introduction section first.")
    else:
        df = st.session_state.df.copy()
        
        if 'Price' not in df.columns:
            st.error("Target variable 'Price' not found in data!")
        else:
            # Prepare features and target
            if 'selected_features' in st.session_state:
                features = st.session_state.selected_features
            else:
                # Use all numeric features except Price
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                features = [col for col in numeric_cols if col != 'Price']
            
            X = df[features]
            y = df['Price']
            
            # Handle missing values in features
            X = X.fillna(X.median())
            
            st.subheader("Data Split")
            test_size = st.slider("Test set size:", 0.1, 0.4, 0.2)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            st.write(f"**Training set:** {X_train.shape[0]} samples")
            st.write(f"**Test set:** {X_test.shape[0]} samples")
            
            # Feature scaling
            st.subheader("Feature Scaling")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Model selection
            st.subheader("Select Models to Train")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                train_lr = st.checkbox("Linear Regression", value=True)
                train_ridge = st.checkbox("Ridge Regression")
                train_lasso = st.checkbox("Lasso Regression")
            
            with col2:
                train_rf = st.checkbox("Random Forest", value=True)
                train_xgb = st.checkbox("XGBoost", value=True)
                train_gbr = st.checkbox("Gradient Boosting")
            
            with col3:
                train_dt = st.checkbox("Decision Tree")
                train_svr = st.checkbox("SVR")
                train_mlp = st.checkbox("Neural Network (MLP)")
            
            # Train models button
            if st.button("🚀 Train Selected Models"):
                st.session_state.models = {}
                st.session_state.results = {}
                
                with st.spinner("Training models..."):
                    # Linear Regression
                    if train_lr:
                        lr = LinearRegression()
                        lr.fit(X_train_scaled, y_train)
                        lr_pred = lr.predict(X_test_scaled)
                        st.session_state.models['Linear Regression'] = lr
                        st.session_state.results['Linear Regression'] = {
                            'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred)),
                            'MAE': mean_absolute_error(y_test, lr_pred),
                            'R2': r2_score(y_test, lr_pred)
                        }
                    
                    # Ridge Regression
                    if train_ridge:
                        ridge = Ridge(alpha=1.0)
                        ridge.fit(X_train_scaled, y_train)
                        ridge_pred = ridge.predict(X_test_scaled)
                        st.session_state.models['Ridge'] = ridge
                        st.session_state.results['Ridge'] = {
                            'RMSE': np.sqrt(mean_squared_error(y_test, ridge_pred)),
                            'MAE': mean_absolute_error(y_test, ridge_pred),
                            'R2': r2_score(y_test, ridge_pred)
                        }
                    
                    # Lasso Regression
                    if train_lasso:
                        lasso = Lasso(alpha=0.01)
                        lasso.fit(X_train_scaled, y_train)
                        lasso_pred = lasso.predict(X_test_scaled)
                        st.session_state.models['Lasso'] = lasso
                        st.session_state.results['Lasso'] = {
                            'RMSE': np.sqrt(mean_squared_error(y_test, lasso_pred)),
                            'MAE': mean_absolute_error(y_test, lasso_pred),
                            'R2': r2_score(y_test, lasso_pred)
                        }
                    
                    # Random Forest
                    if train_rf:
                        rf = RandomForestRegressor(n_estimators=100, random_state=42)
                        rf.fit(X_train_scaled, y_train)
                        rf_pred = rf.predict(X_test_scaled)
                        st.session_state.models['Random Forest'] = rf
                        st.session_state.results['Random Forest'] = {
                            'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
                            'MAE': mean_absolute_error(y_test, rf_pred),
                            'R2': r2_score(y_test, rf_pred)
                        }
                    
                    # XGBoost
                    if train_xgb:
                        xgb = XGBRegressor(n_estimators=100, random_state=42)
                        xgb.fit(X_train_scaled, y_train)
                        xgb_pred = xgb.predict(X_test_scaled)
                        st.session_state.models['XGBoost'] = xgb
                        st.session_state.results['XGBoost'] = {
                            'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred)),
                            'MAE': mean_absolute_error(y_test, xgb_pred),
                            'R2': r2_score(y_test, xgb_pred)
                        }
                    
                    # Gradient Boosting
                    if train_gbr:
                        gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
                        gbr.fit(X_train_scaled, y_train)
                        gbr_pred = gbr.predict(X_test_scaled)
                        st.session_state.models['Gradient Boosting'] = gbr
                        st.session_state.results['Gradient Boosting'] = {
                            'RMSE': np.sqrt(mean_squared_error(y_test, gbr_pred)),
                            'MAE': mean_absolute_error(y_test, gbr_pred),
                            'R2': r2_score(y_test, gbr_pred)
                        }
                    
                    # Decision Tree
                    if train_dt:
                        dt = DecisionTreeRegressor(random_state=42)
                        dt.fit(X_train_scaled, y_train)
                        dt_pred = dt.predict(X_test_scaled)
                        st.session_state.models['Decision Tree'] = dt
                        st.session_state.results['Decision Tree'] = {
                            'RMSE': np.sqrt(mean_squared_error(y_test, dt_pred)),
                            'MAE': mean_absolute_error(y_test, dt_pred),
                            'R2': r2_score(y_test, dt_pred)
                        }
                    
                    # SVR
                    if train_svr:
                        svr = SVR()
                        svr.fit(X_train_scaled, y_train)
                        svr_pred = svr.predict(X_test_scaled)
                        st.session_state.models['SVR'] = svr
                        st.session_state.results['SVR'] = {
                            'RMSE': np.sqrt(mean_squared_error(y_test, svr_pred)),
                            'MAE': mean_absolute_error(y_test, svr_pred),
                            'R2': r2_score(y_test, svr_pred)
                        }
                    
                    # MLP
                    if train_mlp:
                        mlp = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
                        mlp.fit(X_train_scaled, y_train)
                        mlp_pred = mlp.predict(X_test_scaled)
                        st.session_state.models['MLP'] = mlp
                        st.session_state.results['MLP'] = {
                            'RMSE': np.sqrt(mean_squared_error(y_test, mlp_pred)),
                            'MAE': mean_absolute_error(y_test, mlp_pred),
                            'R2': r2_score(y_test, mlp_pred)
                        }
                
                st.success("✅ Models trained successfully!")

# Section 7: Results
elif section == "Results":
    st.header("📊 Model Results")
    
    if not st.session_state.results:
        st.info("Please train models first in the Model Training section.")
    else:
        results = st.session_state.results
        
        # Create results dataframe
        results_df = pd.DataFrame([
            {
                'Model': model,
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'R² Score': metrics['R2']
            }
            for model, metrics in results.items()
        ])
        
        # Sort by R² score
        results_df = results_df.sort_values('R² Score', ascending=False).reset_index(drop=True)
        
        st.subheader("Performance Comparison")
        st.dataframe(results_df.style.format({
            'RMSE': '{:.4f}',
            'MAE': '{:.4f}',
            'R² Score': '{:.4f}'
        }), use_container_width=True)
        
        # Visualizations
        st.subheader("Performance Visualization")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # RMSE Comparison
        axes[0].barh(results_df['Model'], results_df['RMSE'])
        axes[0].set_xlabel('RMSE (Lower is better)')
        axes[0].set_title('RMSE Comparison')
        
        # MAE Comparison
        axes[1].barh(results_df['Model'], results_df['MAE'])
        axes[1].set_xlabel('MAE (Lower is better)')
        axes[1].set_title('MAE Comparison')
        
        # R² Score Comparison
        axes[2].barh(results_df['Model'], results_df['R² Score'])
        axes[2].set_xlabel('R² Score (Higher is better)')
        axes[2].set_title('R² Score Comparison')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Best model
        best_model = results_df.iloc[0]['Model']
        st.success(f"🏆 **Best Model:** {best_model} (R² Score: {results_df.iloc[0]['R² Score']:.4f})")

# Section 8: Predictions
elif section == "Predictions":
    st.header("🔮 Make Predictions")
    
    if not st.session_state.models:
        st.info("Please train models first in the Model Training section.")
    else:
        models = st.session_state.models
        
        # Select model
        selected_model = st.selectbox(
            "Select model for prediction:",
            list(models.keys())
        )
        
        model = models[selected_model]
        
        # Create input form based on features
        if 'selected_features' in st.session_state:
            features = st.session_state.selected_features
            
            st.subheader("Input Features")
            
            # Create columns for inputs
            n_cols = 3
            cols = st.columns(n_cols)
            
            input_values = {}
            for i, feature in enumerate(features):
                with cols[i % n_cols]:
                    # Get statistics for this feature
                    if st.session_state.df is not None:
                        mean_val = st.session_state.df[feature].mean()
                        std_val = st.session_state.df[feature].std()
                        min_val = st.session_state.df[feature].min()
                        max_val = st.session_state.df[feature].max()
                        
                        input_values[feature] = st.number_input(
                            f"{feature}:",
                            value=float(mean_val),
                            min_value=float(min_val * 0.5),
                            max_value=float(max_val * 1.5),
                            step=float(std_val * 0.1)
                        )
            
            # Make prediction
            if st.button("Predict Price"):
                # Prepare input array
                input_array = np.array([input_values[feature] for feature in features]).reshape(1, -1)
                
                # Scale input (assuming StandardScaler was used)
                scaler = StandardScaler()
                if st.session_state.df is not None:
                    X_train = st.session_state.df[features].fillna(
                        st.session_state.df[features].median()
                    )
                    scaler.fit(X_train)
                    input_scaled = scaler.transform(input_array)
                
                # Make prediction
                prediction = model.predict(input_scaled)
                
                st.success(f"**Predicted Price:** ${prediction[0]:.2f}")
                
                # Show confidence interval (simplified)
                st.info("Note: For production use, consider adding confidence intervals and uncertainty estimates.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
This app demonstrates a complete ML pipeline for palm oil price prediction.

**Steps:**
1. Load and preprocess data
2. Perform exploratory analysis
3. Engineer features
4. Train multiple ML models
5. Compare results
6. Make predictions
""")
