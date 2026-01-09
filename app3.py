# app.py - Loads from price.csv
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import os

# Machine Learning libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Palm Oil Price Prediction",
    page_icon="🌴",
    layout="wide"
)

# Custom CSS for better styling
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
        color: #228B22;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .model-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #2E8B57;
    }
    .tuned-model {
        border-left: 5px solid #FF6B6B;
    }
    .stProgress > div > div > div > div {
        background-color: #2E8B57;
    }
    .data-info {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">🌴 Palm Oil Price Prediction System</h1>', unsafe_allow_html=True)
st.markdown("### Focused on Tree-Based Machine Learning Models")

# Initialize session state for data storage
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Choose a section",
    ["📊 Data Overview", "🔧 Data Processing", "🤖 Model Training", "📈 Results & Visualization", "🔮 Make Predictions"]
)

# Function to load data from CSV
def load_data_from_csv(file_path="price.csv"):
    """Load data from CSV file"""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            st.error(f"File '{file_path}' not found. Please make sure the file is in the same directory.")
            return None
        
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Check if 'Date' column exists
        if 'Date' not in df.columns:
            st.error("CSV file must contain a 'Date' column")
            return None
        
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Check if 'Price' column exists
        if 'Price' not in df.columns:
            st.error("CSV file must contain a 'Price' column")
            return None
        
        return df
    
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None

def preprocess_data(df):
    """Preprocess the data"""
    df_processed = df.copy()
    
    # Ensure Date is datetime
    if 'Date' in df_processed.columns:
        df_processed['Date'] = pd.to_datetime(df_processed['Date'])
        df_processed = df_processed.sort_values('Date').reset_index(drop=True)
        
        # Create time-based features
        df_processed['Year'] = df_processed['Date'].dt.year
        df_processed['Month'] = df_processed['Date'].dt.month
        df_processed['Day'] = df_processed['Date'].dt.day
        df_processed['DayOfWeek'] = df_processed['Date'].dt.dayofweek
        df_processed['Quarter'] = df_processed['Date'].dt.quarter
        df_processed['DayOfYear'] = df_processed['Date'].dt.dayofyear
    
    # Handle missing values
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'Price':  # Don't fill target variable
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    return df_processed

def train_tree_models(X_train, X_test, y_train, y_test, tune_models=True):
    """Train tree-based models with optional hyperparameter tuning"""
    results = []
    predictions = {}
    trained_models = {}
    
    # Define models with their hyperparameter grids for tuning
    models_config = {
        'Random Forest': {
            'untuned': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1
            ),
            'tuned_params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'Decision Tree': {
            'untuned': DecisionTreeRegressor(
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42
            ),
            'tuned_params': {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'criterion': ['squared_error', 'friedman_mse', 'absolute_error']
            }
        },
        'XGBoost': {
            'untuned': XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'tuned_params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
        },
        'Gradient Boosting': {
            'untuned': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42
            ),
            'tuned_params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [2, 3, 4, 5],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
    }
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_models = len(models_config) * 2 if tune_models else len(models_config)
    current_model = 0
    
    for model_name, config in models_config.items():
        # Train untuned model
        status_text.text(f"Training {model_name} (Untuned)...")
        
        start_time = time.time()
        untuned_model = config['untuned']
        untuned_model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred_untuned = untuned_model.predict(X_test)
        predictions[f"{model_name} (Untuned)"] = y_pred_untuned
        trained_models[f"{model_name} (Untuned)"] = untuned_model
        
        # Calculate metrics
        rmse_untuned = np.sqrt(mean_squared_error(y_test, y_pred_untuned))
        mae_untuned = mean_absolute_error(y_test, y_pred_untuned)
        r2_untuned = r2_score(y_test, y_pred_untuned)
        
        results.append({
            'Model': f"{model_name} (Untuned)",
            'Type': 'Untuned',
            'Base Model': model_name,
            'RMSE': rmse_untuned,
            'MAE': mae_untuned,
            'R² Score': r2_untuned,
            'Training Time (s)': training_time,
            'Tuned': False
        })
        
        current_model += 1
        progress_bar.progress(current_model / total_models)
        
        # Train tuned model if requested
        if tune_models:
            status_text.text(f"Training {model_name} (Tuned)...")
            
            start_time = time.time()
            
            # Perform grid search
            grid_search = GridSearchCV(
                estimator=config['untuned'],
                param_grid=config['tuned_params'],
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Get best model
            tuned_model = grid_search.best_estimator_
            
            # Make predictions
            y_pred_tuned = tuned_model.predict(X_test)
            predictions[f"{model_name} (Tuned)"] = y_pred_tuned
            trained_models[f"{model_name} (Tuned)"] = tuned_model
            
            # Calculate metrics
            rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
            mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
            r2_tuned = r2_score(y_test, y_pred_tuned)
            
            results.append({
                'Model': f"{model_name} (Tuned)",
                'Type': 'Tuned',
                'Base Model': model_name,
                'RMSE': rmse_tuned,
                'MAE': mae_tuned,
                'R² Score': r2_tuned,
                'Training Time (s)': training_time,
                'Best Params': str(grid_search.best_params_),
                'Tuned': True
            })
            
            current_model += 1
            progress_bar.progress(current_model / total_models)
    
    status_text.text("Training complete!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('R² Score', ascending=False)
    
    return results_df, predictions, trained_models

# Main app logic based on navigation selection
if app_mode == "📊 Data Overview":
    st.markdown('<h2 class="sub-header">Data Overview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Data Source")
        
        # Option to load from CSV
        if st.button("Load from price.csv", type="primary"):
            with st.spinner("Loading data from price.csv..."):
                df = load_data_from_csv("price.csv")
                
                if df is not None:
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    st.success("Data loaded successfully from price.csv!")
                    
                    # Show file info
                    st.markdown('<div class="data-info">', unsafe_allow_html=True)
                    st.write(f"**File:** price.csv")
                    st.write(f"**Rows:** {len(df)}")
                    st.write(f"**Columns:** {len(df.columns)}")
                    st.write(f"**Date Range:** {df['Date'].min().date()} to {df['Date'].max().date()}")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Also keep the option to upload custom file
        st.subheader("Or Upload Custom CSV")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.success("Custom CSV file loaded successfully!")
    
    with col2:
        if st.session_state.data_loaded:
            df = st.session_state.df
            
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Basic statistics
            st.subheader("Data Statistics")
            st.dataframe(df.describe(), use_container_width=True)
            
            # Column information
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.notnull().sum(),
                'Null Count': df.isnull().sum(),
                'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(col_info, use_container_width=True)
            
            # Date range info
            if 'Date' in df.columns:
                st.subheader("Date Information")
                st.write(f"**Start Date:** {df['Date'].min()}")
                st.write(f"**End Date:** {df['Date'].max()}")
                st.write(f"**Total Days:** {(df['Date'].max() - df['Date'].min()).days}")
                st.write(f"**Missing Dates:** Checking...")
                
                # Check for missing dates
                if df['Date'].is_monotonic_increasing:
                    date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max())
                    missing_dates = date_range.difference(df['Date'])
                    st.write(f"**Missing Dates Count:** {len(missing_dates)}")
            
            # Target variable info
            if 'Price' in df.columns:
                st.subheader("Price Statistics")
                price_stats = df['Price'].describe()
                st.write(f"**Mean:** ${price_stats['mean']:,.2f}")
                st.write(f"**Median:** ${price_stats['50%']:,.2f}")
                st.write(f"**Min:** ${price_stats['min']:,.2f}")
                st.write(f"**Max:** ${price_stats['max']:,.2f}")
                st.write(f"**Std Dev:** ${price_stats['std']:,.2f}")
        else:
            st.info("No data loaded yet. Click 'Load from price.csv' to load the data.")

elif app_mode == "🔧 Data Processing":
    st.markdown('<h2 class="sub-header">Data Processing</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first in the 'Data Overview' section.")
        st.info("Click 'Load from price.csv' in the Data Overview section to load your data.")
    else:
        df = st.session_state.df
        
        # Show original data info
        st.subheader("Original Data Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Rows", len(df))
            st.metric("Total Columns", len(df.columns))
        
        with col2:
            if 'Date' in df.columns:
                st.metric("Start Date", df['Date'].min().date())
                st.metric("End Date", df['Date'].max().date())
        
        with col3:
            if 'Price' in df.columns:
                st.metric("Avg Price", f"${df['Price'].mean():,.2f}")
                st.metric("Missing Values", df.isnull().sum().sum())
        
        # Data preprocessing options
        st.subheader("Preprocessing Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Date range selection
            if 'Date' in df.columns:
                min_date = df['Date'].min().date()
                max_date = df['Date'].max().date()
                date_range = st.date_input(
                    "Select Date Range",
                    [min_date, max_date],
                    min_value=min_date,
                    max_value=max_date
                )
            
            # Handle missing values
            missing_method = st.selectbox(
                "Missing Value Handling",
                ["Median Imputation", "Mean Imputation", "Forward Fill", "Backward Fill"]
            )
            
            # Remove outliers
            remove_outliers = st.checkbox("Remove Price Outliers (IQR method)", value=False)
        
        with col2:
            # Feature engineering options
            st.subheader("Feature Engineering")
            create_time_features = st.checkbox("Create Time Features", value=True)
            create_lag_features = st.checkbox("Create Lag Features", value=True)
            create_rolling_features = st.checkbox("Create Rolling Statistics", value=True)
            
            if create_lag_features:
                lag_days = st.multiselect(
                    "Select Lag Days",
                    [1, 2, 3, 7, 14, 30],
                    default=[1, 7, 30]
                )
            
            if create_rolling_features:
                rolling_windows = st.multiselect(
                    "Select Rolling Windows",
                    [3, 7, 14, 30],
                    default=[7, 30]
                )
        
        # Process data button
        if st.button("Process Data", type="primary"):
            with st.spinner("Processing data..."):
                # Preprocess the data
                df_processed = preprocess_data(df)
                
                # Apply date filtering
                if 'Date' in df_processed.columns and len(date_range) == 2:
                    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
                    df_processed = df_processed[
                        (df_processed['Date'] >= start_date) & 
                        (df_processed['Date'] <= end_date)
                    ].copy()
                
                # Handle missing values based on selection
                numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    if col != 'Price' and df_processed[col].isnull().any():
                        if missing_method == "Median Imputation":
                            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                        elif missing_method == "Mean Imputation":
                            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
                        elif missing_method == "Forward Fill":
                            df_processed[col] = df_processed[col].fillna(method='ffill')
                        elif missing_method == "Backward Fill":
                            df_processed[col] = df_processed[col].fillna(method='bfill')
                
                # Remove outliers if selected
                if remove_outliers and 'Price' in df_processed.columns:
                    Q1 = df_processed['Price'].quantile(0.25)
                    Q3 = df_processed['Price'].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    before = len(df_processed)
                    df_processed = df_processed[
                        (df_processed['Price'] >= lower_bound) & 
                        (df_processed['Price'] <= upper_bound)
                    ]
                    after = len(df_processed)
                    st.info(f"Removed {before - after} outliers from Price column")
                
                # Additional feature engineering
                if create_lag_features and 'Price' in df_processed.columns:
                    for lag in lag_days:
                        df_processed[f'Price_Lag_{lag}'] = df_processed['Price'].shift(lag)
                
                if create_rolling_features and 'Price' in df_processed.columns:
                    for window in rolling_windows:
                        df_processed[f'Price_RollingMean_{window}'] = df_processed['Price'].rolling(window=window).mean()
                        df_processed[f'Price_RollingStd_{window}'] = df_processed['Price'].rolling(window=window).std()
                
                # Forward fill any NaN values created by lag/rolling features
                df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
                
                # Drop any remaining NaN values
                df_processed = df_processed.dropna()
                
                # Store processed data
                st.session_state.df_processed = df_processed
                
                st.success("Data processed successfully!")
                
                # Show processed data info
                st.subheader("Processed Data Information")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Processed Rows", len(df_processed))
                    st.metric("Processed Columns", len(df_processed.columns))
                
                with col2:
                    if 'Date' in df_processed.columns:
                        st.metric("Processed Start", df_processed['Date'].min().date())
                        st.metric("Processed End", df_processed['Date'].max().date())
                
                with col3:
                    if 'Price' in df_processed.columns:
                        st.metric("Processed Avg Price", f"${df_processed['Price'].mean():,.2f}")
                        st.metric("Remaining Missing Values", df_processed.isnull().sum().sum())
                
                # Show correlation analysis
                st.subheader("Feature Correlation with Price")
                if 'Price' in df_processed.columns:
                    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
                    correlations = df_processed[numeric_cols].corr()['Price'].sort_values(ascending=False)
                    
                    # Display correlation table
                    corr_df = pd.DataFrame({
                        'Feature': correlations.index,
                        'Correlation': correlations.values
                    }).reset_index(drop=True)
                    
                    # Show top 20 features
                    st.dataframe(corr_df.head(20), use_container_width=True)
                    
                    # Visualize top correlations
                    fig, ax = plt.subplots(figsize=(12, 8))
                    top_corr = correlations.drop('Price' if 'Price' in correlations.index else []).head(15)
                    colors = ['green' if x > 0 else 'red' for x in top_corr.values]
                    bars = ax.barh(range(len(top_corr)), top_corr.values, color=colors)
                    ax.set_yticks(range(len(top_corr)))
                    ax.set_yticklabels(top_corr.index)
                    ax.set_xlabel('Correlation with Price')
                    ax.set_title('Top 15 Features Correlated with Price')
                    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                    
                    # Add correlation values on bars
                    for i, (bar, corr) in enumerate(zip(bars, top_corr.values)):
                        ax.text(bar.get_width() + (0.01 if corr >= 0 else -0.05), 
                               bar.get_y() + bar.get_height()/2,
                               f'{corr:.3f}', 
                               va='center',
                               ha='left' if corr >= 0 else 'right',
                               color='black')
                    
                    st.pyplot(fig)

# ... (The rest of the code for Model Training, Results & Visualization, and Make Predictions remains the same)
# Note: You'll need to copy the rest of the code from the previous version for these sections

# For brevity, I'll show the continuation but you should keep all the existing code
# from the previous version for the remaining sections:
elif app_mode == "🤖 Model Training":
    # [Keep all the Model Training code from previous version]
    pass

elif app_mode == "📈 Results & Visualization":
    # [Keep all the Results & Visualization code from previous version]
    pass

elif app_mode == "🔮 Make Predictions":
    # [Keep all the Make Predictions code from previous version]
    pass

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🌴 <b>Palm Oil Price Prediction System</b> | CSM1 Group Project | BSD3523 Machine Learning</p>
    <p>Focused on Tree-Based Models: Random Forest, Decision Tree, XGBoost, Gradient Boosting</p>
    <p>Data loaded from: price.csv</p>
    <p>Group Members: YIP YOONG ENG, MUHAMMAS AMIRUL AMIER, ALIYA AFIFAH, NUR IZZATI, ALIA AYUNNI</p>
</div>
""", unsafe_allow_html=True)
