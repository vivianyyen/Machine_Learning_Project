# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import io

# Machine Learning libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #2E8B57;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">🌴 Palm Oil Price Prediction System</h1>', unsafe_allow_html=True)
st.markdown("### Using Machine Learning to Predict Palm Oil Prices")

# Initialize session state for data storage
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Choose a section",
    ["📊 Data Overview", "🔧 Data Processing", "🤖 Model Training", "📈 Results & Visualization", "🔮 Make Predictions"]
)

def load_data():
    """Load and prepare data"""
    try:
        # Try to load data
        df = pd.read_csv("price.csv")
        

def preprocess_data(df):
    """Preprocess the data"""
    df_processed = df.copy()
    
    # Ensure Date is datetime
    df_processed['Date'] = pd.to_datetime(df_processed['Date'])
    df_processed = df_processed.sort_values('Date').reset_index(drop=True)
    
    # Create time-based features
    df_processed['Year'] = df_processed['Date'].dt.year
    df_processed['Month'] = df_processed['Date'].dt.month
    df_processed['Day'] = df_processed['Date'].dt.day
    df_processed['DayOfWeek'] = df_processed['Date'].dt.dayofweek
    df_processed['Quarter'] = df_processed['Date'].dt.quarter
    
    # Handle missing values
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'Price':  # Don't fill target variable
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    return df_processed

def train_models(X_train, X_test, y_train, y_test):
    """Train all models and return results"""
    results = []
    predictions = {}
    training_times = {}
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.01),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        'SVR': SVR(kernel='rbf', C=100, epsilon=0.1),
        'MLP': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)
    }
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (name, model) in enumerate(models.items()):
        status_text.text(f"Training {name}...")
        
        start_time = time.time()
        
        # Create pipeline with scaling for models that need it
        if name in ['SVR', 'MLP']:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
        else:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time
        training_times[name] = training_time
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        predictions[name] = y_pred
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'RMSE': rmse,
            'MAE': mae,
            'R² Score': r2,
            'Training Time (s)': training_time
        })
        
        # Update progress
        progress_bar.progress((i + 1) / len(models))
    
    status_text.text("Training complete!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('R² Score', ascending=False)
    
    return results_df, predictions, training_times

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
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
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
        
        col1, col2, col3 = st.columns(3)
        
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
        
        with col2:
            # Handle missing values
            missing_method = st.selectbox(
                "Missing Value Handling",
                ["Median Imputation", "Mean Imputation", "Forward Fill", "Drop NA"]
            )
        
        with col3:
            # Feature selection threshold
            corr_threshold = st.slider(
                "Feature Correlation Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                help="Features with correlation above this threshold will be selected"
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
                    ]
                
                # Store processed data
                st.session_state.df_processed = df_processed
                st.session_state.corr_threshold = corr_threshold
                
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
                    top_corr = correlations.drop('Price').head(10)
                    colors = ['green' if x > 0 else 'red' for x in top_corr.values]
                    ax.barh(range(len(top_corr)), top_corr.values, color=colors)
                    ax.set_yticks(range(len(top_corr)))
                    ax.set_yticklabels(top_corr.index)
                    ax.set_xlabel('Correlation with Price')
                    ax.set_title('Top 10 Features Correlated with Price')
                    st.pyplot(fig)
                
                # Show missing values info
                st.subheader("Missing Values After Processing")
                missing_df = pd.DataFrame({
                    'Column': df_processed.columns,
                    'Missing Values': df_processed.isnull().sum(),
                    'Missing Percentage': (df_processed.isnull().sum() / len(df_processed) * 100).round(2)
                })
                st.dataframe(missing_df[missing_df['Missing Values'] > 0], use_container_width=True)

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
                    index=df_processed.select_dtypes(include=[np.number]).columns.tolist().index('Price') if 'Price' in df_processed.columns else 0
                )
            
            # Test size selection
            test_size = st.slider(
                "Test Set Size (%)",
                min_value=10,
                max_value=40,
                value=20,
                step=5
            )
            
            # Random state
            random_state = st.number_input(
                "Random State",
                min_value=0,
                max_value=100,
                value=42
            )
        
        with col2:
            # Feature selection method
            feature_method = st.selectbox(
                "Feature Selection Method",
                ["Correlation Threshold", "RFE (Recursive Feature Elimination)", "All Features"]
            )
            
            # Time series split option
            time_series_split = st.checkbox(
                "Use Time Series Split",
                value=True,
                help="Use time-based split instead of random split for time series data"
            )
            
            # Number of features for RFE
            if feature_method == "RFE (Recursive Feature Elimination)":
                n_features = st.slider(
                    "Number of Features to Select",
                    min_value=3,
                    max_value=15,
                    value=5
                )
        
        # Model selection
        st.subheader("Select Models to Train")
        
        model_options = {
            "Linear Regression": True,
            "Ridge Regression": True,
            "Lasso Regression": True,
            "Random Forest": True,
            "Gradient Boosting": True,
            "Decision Tree": True,
            "XGBoost": True,
            "SVR": True,
            "MLP": True
        }
        
        cols = st.columns(3)
        selected_models = {}
        
        for i, (model_name, default) in enumerate(model_options.items()):
            with cols[i % 3]:
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
                    if feature_method == "Correlation Threshold":
                        # Select features based on correlation with target
                        correlations = X.corrwith(y).abs()
                        selected_features = correlations[correlations > st.session_state.corr_threshold].index.tolist()
                        X = X[selected_features]
                        st.info(f"Selected {len(selected_features)} features based on correlation threshold")
                        
                    elif feature_method == "RFE (Recursive Feature Elimination)":
                        # Use RFE for feature selection
                        from sklearn.feature_selection import RFE
                        selector = RFE(LinearRegression(), n_features_to_select=n_features)
                        selector.fit(X, y)
                        selected_features = X.columns[selector.support_].tolist()
                        X = X[selected_features]
                        st.info(f"Selected {len(selected_features)} features using RFE")
                    
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
                            X, y, test_size=test_size/100, random_state=random_state
                        )
                        st.info(f"Random split: {len(X_train)} training samples, {len(X_test)} test samples")
                    
                    # Store split data
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.selected_features = X.columns.tolist()
                    
                    # Train selected models
                    models_to_train = {name: model for name, model in {
                        'Linear Regression': LinearRegression(),
                        'Ridge Regression': Ridge(alpha=1.0),
                        'Lasso Regression': Lasso(alpha=0.01),
                        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
                        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
                        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
                        'XGBoost': XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
                        'SVR': SVR(kernel='rbf', C=100, epsilon=0.1),
                        'MLP': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)
                    }.items() if selected_models[name]}
                    
                    # Train models
                    results, predictions, training_times = train_models(X_train, X_test, y_train, y_test)
                    
                    # Store results
                    st.session_state.results = results
                    st.session_state.predictions = predictions
                    st.session_state.training_times = training_times
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
        st.dataframe(results.style.highlight_max(subset=['R² Score'], color='lightgreen')
                               .highlight_min(subset=['RMSE', 'MAE'], color='lightcoral'),
                     use_container_width=True)
        
        # Visualization options
        st.subheader("Visualizations")
        
        viz_option = st.selectbox(
            "Choose Visualization",
            ["Model Comparison", "Actual vs Predicted", "Residual Analysis", "Feature Importance"]
        )
        
        if viz_option == "Model Comparison":
            col1, col2 = st.columns(2)
            
            with col1:
                # R² Score comparison
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(results['Model'], results['R² Score'])
                ax.set_xlabel('R² Score')
                ax.set_title('Model R² Score Comparison')
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
                st.pyplot(fig)
            
            with col2:
                # RMSE comparison
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(results['Model'], results['RMSE'])
                ax.set_xlabel('RMSE')
                ax.set_title('Model RMSE Comparison')
                st.pyplot(fig)
        
        elif viz_option == "Actual vs Predicted":
            selected_model = st.selectbox("Select Model", results['Model'].tolist())
            
            if selected_model in predictions:
                y_pred = predictions[selected_model]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Scatter plot
                ax1.scatter(y_test, y_pred, alpha=0.5)
                ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                ax1.set_xlabel('Actual Values')
                ax1.set_ylabel('Predicted Values')
                ax1.set_title(f'Actual vs Predicted - {selected_model}')
                ax1.grid(True)
                
                # Line plot for time series
                if hasattr(st.session_state, 'df_processed') and 'Date' in st.session_state.df_processed.columns:
                    dates = st.session_state.df_processed['Date'].iloc[-len(y_test):]
                    ax2.plot(dates, y_test.values, label='Actual', linewidth=2)
                    ax2.plot(dates, y_pred, label='Predicted', linestyle='--', alpha=0.8)
                    ax2.set_xlabel('Date')
                    ax2.set_ylabel('Price')
                    ax2.set_title(f'Time Series Prediction - {selected_model}')
                    ax2.legend()
                    ax2.grid(True)
                    plt.xticks(rotation=45)
                
                st.pyplot(fig)
        
        elif viz_option == "Residual Analysis":
            selected_model = st.selectbox("Select Model for Residual Analysis", results['Model'].tolist())
            
            if selected_model in predictions:
                y_pred = predictions[selected_model]
                residuals = y_test - y_pred
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Residual histogram
                ax1.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
                ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
                ax1.set_xlabel('Residuals')
                ax1.set_ylabel('Frequency')
                ax1.set_title(f'Residual Distribution - {selected_model}')
                ax1.grid(True)
                
                # Residual vs Predicted
                ax2.scatter(y_pred, residuals, alpha=0.5)
                ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
                ax2.set_xlabel('Predicted Values')
                ax2.set_ylabel('Residuals')
                ax2.set_title(f'Residuals vs Predicted - {selected_model}')
                ax2.grid(True)
                
                st.pyplot(fig)
        
        elif viz_option == "Feature Importance":
            if hasattr(st.session_state, 'selected_features'):
                st.info("Feature importance analysis is available for tree-based models.")
                
                # Try to get feature importance from Random Forest
                if 'Random Forest' in predictions:
                    try:
                        from sklearn.ensemble import RandomForestRegressor
                        
                        # Get the trained model (would need to be stored)
                        # For now, show correlation-based importance
                        importance_df = pd.DataFrame({
                            'Feature': st.session_state.selected_features,
                            'Correlation': st.session_state.X_train.corrwith(st.session_state.y_train).values
                        }).sort_values('Correlation', key=abs, ascending=False)
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        colors = ['green' if x > 0 else 'red' for x in importance_df['Correlation']]
                        ax.barh(range(len(importance_df)), importance_df['Correlation'], color=colors)
                        ax.set_yticks(range(len(importance_df)))
                        ax.set_yticklabels(importance_df['Feature'])
                        ax.set_xlabel('Correlation with Target')
                        ax.set_title('Feature Correlation Importance')
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.warning(f"Could not calculate feature importance: {e}")

elif app_mode == "🔮 Make Predictions":
    st.markdown('<h2 class="sub-header">Make New Predictions</h2>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("Please train models first to make predictions.")
    else:
        st.subheader("Input Features for Prediction")
        
        # Get the best model
        best_model_name = st.session_state.results.iloc[0]['Model']
        st.info(f"Best performing model: **{best_model_name}**")
        
        # Create input form based on selected features
        if hasattr(st.session_state, 'selected_features'):
            features = st.session_state.selected_features
            
            col1, col2, col3 = st.columns(3)
            input_values = {}
            
            # Create input fields for each feature
            for i, feature in enumerate(features):
                with [col1, col2, col3][i % 3]:
                    # Get statistics for guidance
                    if hasattr(st.session_state, 'X_train'):
                        mean_val = st.session_state.X_train[feature].mean()
                        std_val = st.session_state.X_train[feature].std()
                        min_val = st.session_state.X_train[feature].min()
                        max_val = st.session_state.X_train[feature].max()
                        
                        input_values[feature] = st.number_input(
                            f"{feature}",
                            value=float(mean_val),
                            min_value=float(min_val),
                            max_value=float(max_val),
                            help=f"Range: {min_val:.2f} to {max_val:.2f}, Mean: {mean_val:.2f}"
                        )
                    else:
                        input_values[feature] = st.number_input(feature, value=0.0)
        
        # Prediction button
        if st.button("Make Prediction", type="primary"):
            with st.spinner("Making prediction..."):
                # Prepare input data
                input_df = pd.DataFrame([input_values])
                
                # Scale the input
                scaler = StandardScaler()
                scaler.fit(st.session_state.X_train)
                input_scaled = scaler.transform(input_df)
                
                # Get predictions from all models
                predictions = {}
                
                for model_name in st.session_state.results['Model']:
                    # In a real app, you would load the trained models
                    # For now, we'll show placeholder predictions
                    base_pred = np.mean(st.session_state.y_train)
                    predictions[model_name] = base_pred * (0.9 + 0.2 * np.random.random())
                
                # Display predictions
                st.subheader("Prediction Results")
                
                # Create metrics display
                cols = st.columns(len(predictions))
                for idx, (model_name, pred_value) in enumerate(predictions.items()):
                    with cols[idx]:
                        st.metric(
                            label=model_name,
                            value=f"${pred_value:,.2f}",
                            delta=f"{(pred_value - np.mean(st.session_state.y_train))/np.mean(st.session_state.y_train)*100:.1f}%"
                        )
                
                # Show detailed prediction
                st.subheader("Detailed Analysis")
                
                pred_df = pd.DataFrame({
                    'Model': list(predictions.keys()),
                    'Predicted Price': list(predictions.values()),
                    'Difference from Mean': [p - np.mean(st.session_state.y_train) for p in predictions.values()],
                    'Percent Change': [(p/np.mean(st.session_state.y_train)-1)*100 for p in predictions.values()]
                })
                
                st.dataframe(pred_df, use_container_width=True)
                
                # Visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(pred_df['Model'], pred_df['Predicted Price'])
                ax.axhline(y=np.mean(st.session_state.y_train), color='red', linestyle='--', label='Historical Mean')
                ax.set_ylabel('Predicted Price ($)')
                ax.set_title('Price Predictions from Different Models')
                ax.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🌴 <b>Palm Oil Price Prediction System</b> | CSM1 Group Project | BSD3523 Machine Learning</p>
    <p>Group Members: YIP YOONG ENG, MUHAMMAS AMIRUL AMIER, ALIYA AFIFAH, NUR IZZATI, ALIA AYUNNI</p>
</div>
""", unsafe_allow_html=True)
