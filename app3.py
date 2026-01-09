# app.py - Focused on Tree-Based Models
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import io
import joblib

# Machine Learning libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

@st.cache_data
def load_data():
    """Load and prepare data"""
    try:
        # Try to load data
        df = pd.read_csv("price.csv")
    return df


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
                random_state=42
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
                random_state=42
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
        st.subheader("Upload Your Data")
        upload_option = st.radio(
            "Choose data source:",
            ["Use Sample Data", "Upload CSV File"]
        )
        
        if upload_option == "Upload CSV File":
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success("Data loaded successfully!")
            else:
                st.info("Please upload a CSV file or use sample data.")
        else:
            if st.button("Load Sample Data"):
                df = load_sample_data()
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success("Sample data loaded successfully!")
    
    with col2:
        if st.session_state.data_loaded:
            df = st.session_state.df
            
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Basic statistics
            st.subheader("Data Statistics")
            st.dataframe(df.describe(), use_container_width=True)
            
            # Data info
            st.subheader("Data Information")
            st.write(f"Shape: {df.shape}")
            st.write(f"Columns: {list(df.columns)}")
            st.write(f"Date Range: {df['Date'].min() if 'Date' in df.columns else 'N/A'} to {df['Date'].max() if 'Date' in df.columns else 'N/A'}")
        else:
            st.info("No data loaded yet. Please load data from the left panel.")

elif app_mode == "🔧 Data Processing":
    st.markdown('<h2 class="sub-header">Data Processing</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first in the 'Data Overview' section.")
        if st.button("Load Sample Data Now"):
            df = load_sample_data()
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.rerun()
    else:
        df = st.session_state.df
        
        # Data preprocessing options
        st.subheader("Preprocessing Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Date range selection
            if 'Date' in df.columns:
                min_date = pd.to_datetime(df['Date']).min()
                max_date = pd.to_datetime(df['Date']).max()
                date_range = st.date_input(
                    "Select Date Range",
                    [min_date, max_date],
                    min_value=min_date,
                    max_value=max_date
                )
            
            # Handle missing values
            missing_method = st.selectbox(
                "Missing Value Handling",
                ["Median Imputation", "Mean Imputation", "Forward Fill"]
            )
        
        with col2:
            # Feature engineering options
            st.subheader("Feature Engineering")
            create_lag_features = st.checkbox("Create Lag Features", value=True)
            create_rolling_features = st.checkbox("Create Rolling Statistics", value=True)
            create_time_features = st.checkbox("Create Time Features", value=True)
        
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
                    ]
                
                # Additional feature engineering
                if create_lag_features and 'Price' in df_processed.columns:
                    df_processed['Price_Lag1'] = df_processed['Price'].shift(1)
                    df_processed['Price_Lag7'] = df_processed['Price'].shift(7)
                    df_processed['Price_Lag30'] = df_processed['Price'].shift(30)
                
                if create_rolling_features and 'Price' in df_processed.columns:
                    df_processed['Price_RollingMean_7'] = df_processed['Price'].rolling(window=7).mean()
                    df_processed['Price_RollingStd_7'] = df_processed['Price'].rolling(window=7).std()
                
                # Forward fill any NaN values created by lag features
                df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
                
                # Store processed data
                st.session_state.df_processed = df_processed
                
                st.success("Data processed successfully!")
                
                # Show processed data info
                st.subheader("Processed Data Preview")
                st.dataframe(df_processed.head(), use_container_width=True)
                
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
                    
                    st.dataframe(corr_df, use_container_width=True)
                    
                    # Visualize correlations
                    fig, ax = plt.subplots(figsize=(10, 6))
                    top_corr = correlations.drop('Price' if 'Price' in correlations.index else []).head(15)
                    colors = ['green' if x > 0 else 'red' for x in top_corr.values]
                    ax.barh(range(len(top_corr)), top_corr.values, color=colors)
                    ax.set_yticks(range(len(top_corr)))
                    ax.set_yticklabels(top_corr.index)
                    ax.set_xlabel('Correlation with Price')
                    ax.set_title('Top Features Correlated with Price')
                    st.pyplot(fig)

elif app_mode == "🤖 Model Training":
    st.markdown('<h2 class="sub-header">Model Training</h2>', unsafe_allow_html=True)
    
    if 'df_processed' not in st.session_state:
        st.warning("Please process data first in the 'Data Processing' section.")
    else:
        df_processed = st.session_state.df_processed
        
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Target variable selection
            if 'Price' in df_processed.columns:
                target_var = st.selectbox(
                    "Select Target Variable",
                    df_processed.select_dtypes(include=[np.number]).columns.tolist(),
                    index=df_processed.select_dtypes(include=[np.number]).columns.tolist().index('Price')
                )
            
            # Test size selection
            test_size = st.slider(
                "Test Set Size (%)",
                min_value=10,
                max_value=40,
                value=20,
                step=5
            )
            
            # Time series split option
            time_series_split = st.checkbox(
                "Use Time Series Split",
                value=True,
                help="Use time-based split instead of random split for time series data"
            )
        
        with col2:
            # Feature selection
            st.subheader("Feature Selection")
            use_all_features = st.checkbox("Use All Features", value=True)
            
            if not use_all_features:
                corr_threshold = st.slider(
                    "Minimum Correlation Threshold",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.1,
                    step=0.05
                )
            
            # Hyperparameter tuning
            tune_models = st.checkbox(
                "Perform Hyperparameter Tuning",
                value=True,
                help="Enable Grid Search for hyperparameter optimization"
            )
        
        # Model selection
        st.subheader("Select Models to Train")
        
        model_options = {
            "Random Forest": True,
            "Decision Tree": True,
            "XGBoost": True,
            "Gradient Boosting": True
        }
        
        cols = st.columns(4)
        selected_models = {}
        
        for i, (model_name, default) in enumerate(model_options.items()):
            with cols[i]:
                selected_models[model_name] = st.checkbox(model_name, value=default)
        
        # Train models button
        if st.button("Train Models", type="primary", disabled=not any(selected_models.values())):
            if not any(selected_models.values()):
                st.error("Please select at least one model to train.")
            else:
                with st.spinner("Training models..."):
                    # Prepare features and target
                    X = df_processed.drop(columns=['Date', target_var] if 'Date' in df_processed.columns else [target_var])
                    y = df_processed[target_var]
                    
                    # Select only numeric columns
                    X = X.select_dtypes(include=[np.number])
                    
                    # Handle missing values in X
                    X = X.fillna(X.median())
                    
                    # Feature selection
                    if not use_all_features:
                        # Select features based on correlation with target
                        correlations = X.corrwith(y).abs()
                        selected_features = correlations[correlations > corr_threshold].index.tolist()
                        if selected_features:
                            X = X[selected_features]
                            st.info(f"Selected {len(selected_features)} features based on correlation threshold")
                        else:
                            st.warning("No features selected with given threshold. Using all features.")
                    
                    # Split data
                    if time_series_split and 'Date' in df_processed.columns:
                        # Time-based split
                        split_idx = int(len(df_processed) * (1 - test_size/100))
                        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                        st.info(f"Time-based split: {len(X_train)} training samples, {len(X_test)} test samples")
                    else:
                        # Random split
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size/100, random_state=42
                        )
                        st.info(f"Random split: {len(X_train)} training samples, {len(X_test)} test samples")
                    
                    # Store split data
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.selected_features = X.columns.tolist()
                    
                    # Filter models based on selection
                    models_to_train = {name: True for name in selected_models if selected_models[name]}
                    
                    # Train models
                    results, predictions, trained_models = train_tree_models(
                        X_train, X_test, y_train, y_test, tune_models
                    )
                    
                    # Filter results to only include selected models
                    filtered_results = results[results['Base Model'].isin(models_to_train.keys())]
                    
                    # Store results
                    st.session_state.results = filtered_results
                    st.session_state.predictions = predictions
                    st.session_state.trained_models = trained_models
                    st.session_state.models_trained = True
                    
                    st.success("All models trained successfully!")

elif app_mode == "📈 Results & Visualization":
    st.markdown('<h2 class="sub-header">Results & Visualization</h2>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("Please train models first in the 'Model Training' section.")
    else:
        results = st.session_state.results
        predictions = st.session_state.predictions
        y_test = st.session_state.y_test
        
        # Display results table
        st.subheader("Model Performance Comparison")
        
        # Format results for display
        display_results = results.copy()
        if 'Best Params' in display_results.columns:
            display_results = display_results.drop(columns=['Best Params'])
        
        # Create styled dataframe
        st.dataframe(
            display_results.style
            .background_gradient(subset=['R² Score'], cmap='RdYlGn')
            .background_gradient(subset=['RMSE', 'MAE'], cmap='RdYlGn_r')
            .format({
                'RMSE': '{:.2f}',
                'MAE': '{:.2f}',
                'R² Score': '{:.4f}',
                'Training Time (s)': '{:.2f}'
            }),
            use_container_width=True,
            height=400
        )
        
        # Download results button
        csv = results.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="model_results.csv",
            mime="text/csv"
        )
        
        # Visualization options
        st.subheader("Visualizations")
        
        viz_option = st.selectbox(
            "Choose Visualization",
            ["Model Comparison", "Actual vs Predicted", "Feature Importance", "Error Analysis"]
        )
        
        if viz_option == "Model Comparison":
            col1, col2 = st.columns(2)
            
            with col1:
                # R² Score comparison
                fig, ax = plt.subplots(figsize=(10, 6))
                models = results['Model']
                r2_scores = results['R² Score']
                
                # Color based on tuned/untuned
                colors = ['#FF6B6B' if 'Tuned' in model else '#2E8B57' for model in models]
                
                bars = ax.barh(models, r2_scores, color=colors)
                ax.set_xlabel('R² Score')
                ax.set_title('Model R² Score Comparison')
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
                ax.set_xlim(0, max(1.0, max(r2_scores) * 1.1))
                
                # Add value labels
                for bar, score in zip(bars, r2_scores):
                    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{score:.4f}', va='center')
                
                st.pyplot(fig)
            
            with col2:
                # RMSE comparison
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Color based on tuned/untuned
                colors = ['#FF6B6B' if 'Tuned' in model else '#2E8B57' for model in models]
                
                bars = ax.barh(models, results['RMSE'], color=colors)
                ax.set_xlabel('RMSE')
                ax.set_title('Model RMSE Comparison')
                
                # Add value labels
                for bar, rmse in zip(bars, results['RMSE']):
                    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{rmse:.2f}', va='center')
                
                st.pyplot(fig)
        
        elif viz_option == "Actual vs Predicted":
            selected_model = st.selectbox("Select Model", results['Model'].tolist())
            
            if selected_model in predictions:
                y_pred = predictions[selected_model]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Scatter plot
                ax1.scatter(y_test, y_pred, alpha=0.5)
                ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                ax1.set_xlabel('Actual Price')
                ax1.set_ylabel('Predicted Price')
                ax1.set_title(f'Actual vs Predicted - {selected_model}\nR² = {results[results["Model"]==selected_model]["R² Score"].values[0]:.4f}')
                ax1.grid(True)
                
                # Line plot for time series
                if hasattr(st.session_state, 'df_processed') and 'Date' in st.session_state.df_processed.columns:
                    dates = st.session_state.df_processed['Date'].iloc[-len(y_test):]
                    ax2.plot(dates, y_test.values, label='Actual', linewidth=2, color='blue')
                    ax2.plot(dates, y_pred, label='Predicted', linestyle='--', alpha=0.8, color='orange')
                    ax2.set_xlabel('Date')
                    ax2.set_ylabel('Price')
                    ax2.set_title(f'Time Series Prediction - {selected_model}')
                    ax2.legend()
                    ax2.grid(True)
                    plt.xticks(rotation=45)
                
                st.pyplot(fig)
        
        elif viz_option == "Feature Importance":
            selected_model = st.selectbox("Select Model for Feature Importance", 
                                         [m for m in results['Model'].tolist() if 'Random Forest' in m or 'XGBoost' in m or 'Gradient Boosting' in m])
            
            if selected_model in st.session_state.trained_models:
                model = st.session_state.trained_models[selected_model]
                features = st.session_state.selected_features
                
                # Get feature importance
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    
                    # Create feature importance dataframe
                    importance_df = pd.DataFrame({
                        'Feature': features[:len(importances)],
                        'Importance': importances
                    }).sort_values('Importance', ascending=False).head(15)
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 8))
                    bars = ax.barh(range(len(importance_df)), importance_df['Importance'])
                    ax.set_yticks(range(len(importance_df)))
                    ax.set_yticklabels(importance_df['Feature'])
                    ax.set_xlabel('Feature Importance')
                    ax.set_title(f'Feature Importance - {selected_model}')
                    ax.invert_yaxis()
                    
                    # Add value labels
                    for i, (bar, imp) in enumerate(zip(bars, importance_df['Importance'])):
                        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                               f'{imp:.4f}', va='center')
                    
                    st.pyplot(fig)
                    
                    # Display importance table
                    st.dataframe(importance_df, use_container_width=True)
                else:
                    st.warning(f"{selected_model} does not have feature_importances_ attribute")
            else:
                st.warning("Please train the model first to view feature importance")
        
        elif viz_option == "Error Analysis":
            selected_model = st.selectbox("Select Model for Error Analysis", results['Model'].tolist())
            
            if selected_model in predictions:
                y_pred = predictions[selected_model]
                errors = y_test - y_pred
                
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                
                # Error distribution
                ax1.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
                ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
                ax1.set_xlabel('Prediction Error')
                ax1.set_ylabel('Frequency')
                ax1.set_title(f'Error Distribution - {selected_model}')
                ax1.grid(True)
                
                # QQ plot for normality check
                from scipy import stats
                stats.probplot(errors, dist="norm", plot=ax2)
                ax2.set_title(f'Q-Q Plot - {selected_model}')
                ax2.grid(True)
                
                # Error vs Predicted
                ax3.scatter(y_pred, errors, alpha=0.5, color='green')
                ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
                ax3.set_xlabel('Predicted Values')
                ax3.set_ylabel('Errors')
                ax3.set_title(f'Errors vs Predicted - {selected_model}')
                ax3.grid(True)
                
                # Error over time
                if hasattr(st.session_state, 'df_processed') and 'Date' in st.session_state.df_processed.columns:
                    dates = st.session_state.df_processed['Date'].iloc[-len(errors):]
                    ax4.plot(dates, errors, color='purple', alpha=0.7)
                    ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
                    ax4.set_xlabel('Date')
                    ax4.set_ylabel('Prediction Error')
                    ax4.set_title(f'Error Over Time - {selected_model}')
                    ax4.grid(True)
                    plt.xticks(rotation=45)
                
                st.pyplot(fig)

elif app_mode == "🔮 Make Predictions":
    st.markdown('<h2 class="sub-header">Make New Predictions</h2>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("Please train models first to make predictions.")
    else:
        # Get the best model
        if st.session_state.results is not None and not st.session_state.results.empty:
            best_model_name = st.session_state.results.iloc[0]['Model']
            st.info(f"Best performing model: **{best_model_name}**")
        else:
            st.error("No results available. Please train models first.")
            st.stop()
        
        st.subheader("Input Features for Prediction")
        
        # Create input form based on selected features
        if hasattr(st.session_state, 'selected_features'):
            features = st.session_state.selected_features
            
            if len(features) > 0:
                # Create 3 columns for feature inputs
                cols = st.columns(3)
                input_values = {}
                
                # Get statistics for guidance
                if hasattr(st.session_state, 'X_train'):
                    X_train = st.session_state.X_train
                    
                    # Create input fields for each feature
                    for i, feature in enumerate(features):
                        with cols[i % 3]:
                            if feature in X_train.columns:
                                mean_val = X_train[feature].mean()
                                std_val = X_train[feature].std()
                                min_val = X_train[feature].min()
                                max_val = X_train[feature].max()
                                
                                input_values[feature] = st.number_input(
                                    f"{feature}",
                                    value=float(mean_val),
                                    min_value=float(min_val * 0.5),
                                    max_value=float(max_val * 1.5),
                                    help=f"Range: {min_val:.2f} to {max_val:.2f}"
                                )
                            else:
                                input_values[feature] = st.number_input(feature, value=0.0)
                
                # Model selection for prediction
                st.subheader("Select Model for Prediction")
                available_models = list(st.session_state.trained_models.keys())
                selected_model_for_pred = st.selectbox(
                    "Choose model:",
                    available_models,
                    index=0 if best_model_name in available_models else 0
                )
                
                # Prediction button
                if st.button("Make Prediction", type="primary"):
                    with st.spinner("Making prediction..."):
                        # Prepare input data
                        input_df = pd.DataFrame([input_values])
                        
                        # Get the trained model
                        model = st.session_state.trained_models[selected_model_for_pred]
                        
                        # Make prediction
                        try:
                            prediction = model.predict(input_df)[0]
                            
                            # Display prediction
                            st.subheader("Prediction Result")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    label="Predicted Price",
                                    value=f"${prediction:,.2f}",
                                    help=f"Prediction from {selected_model_for_pred}"
                                )
                            
                            with col2:
                                if hasattr(st.session_state, 'y_train'):
                                    mean_price = np.mean(st.session_state.y_train)
                                    diff = prediction - mean_price
                                    st.metric(
                                        label="Difference from Historical Mean",
                                        value=f"${diff:,.2f}",
                                        delta=f"{diff/mean_price*100:.1f}%"
                                    )
                            
                            with col3:
                                if hasattr(st.session_state, 'y_test') and len(st.session_state.y_test) > 0:
                                    test_mean = np.mean(st.session_state.y_test)
                                    diff_test = prediction - test_mean
                                    st.metric(
                                        label="Difference from Test Mean",
                                        value=f"${diff_test:,.2f}",
                                        delta=f"{diff_test/test_mean*100:.1f}%"
                                    )
                            
                            # Additional information
                            st.subheader("Model Information")
                            model_info = st.session_state.results[
                                st.session_state.results['Model'] == selected_model_for_pred
                            ].iloc[0]
                            
                            info_col1, info_col2 = st.columns(2)
                            
                            with info_col1:
                                st.write(f"**Model Type:** {model_info['Base Model']}")
                                st.write(f"**Tuned:** {'Yes' if model_info['Tuned'] else 'No'}")
                                st.write(f"**R² Score:** {model_info['R² Score']:.4f}")
                            
                            with info_col2:
                                st.write(f"**RMSE:** {model_info['RMSE']:.2f}")
                                st.write(f"**MAE:** {model_info['MAE']:.2f}")
                                st.write(f"**Training Time:** {model_info['Training Time (s)']:.2f}s")
                            
                            # Feature importance visualization for this prediction
                            if hasattr(model, 'feature_importances_'):
                                st.subheader("Feature Contribution Analysis")
                                
                                importances = model.feature_importances_
                                contrib_df = pd.DataFrame({
                                    'Feature': features[:len(importances)],
                                    'Importance': importances,
                                    'Value': [input_values.get(f, 0) for f in features[:len(importances)]]
                                }).sort_values('Importance', ascending=False).head(10)
                                
                                # Create horizontal bar chart
                                fig, ax = plt.subplots(figsize=(10, 6))
                                y_pos = np.arange(len(contrib_df))
                                ax.barh(y_pos, contrib_df['Importance'])
                                ax.set_yticks(y_pos)
                                ax.set_yticklabels(contrib_df['Feature'])
                                ax.set_xlabel('Feature Importance')
                                ax.set_title('Top 10 Features Contributing to Prediction')
                                ax.invert_yaxis()
                                
                                st.pyplot(fig)
                        
                        except Exception as e:
                            st.error(f"Error making prediction: {str(e)}")
            else:
                st.warning("No features available for prediction. Please process and train models first.")
        else:
            st.warning("No features selected. Please train models first.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🌴 <b>Palm Oil Price Prediction System</b> | CSM1 Group Project | BSD3523 Machine Learning</p>
    <p>Focused on Tree-Based Models: Random Forest, Decision Tree, XGBoost, Gradient Boosting</p>
    <p>Group Members: YIP YOONG ENG, MUHAMMAS AMIRUL AMIER, ALIYA AFIFAH, NUR IZZATI, ALIA AYUNNI</p>
</div>
""", unsafe_allow_html=True)
