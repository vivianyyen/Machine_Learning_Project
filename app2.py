# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
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
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🌴 Palm Oil Price Prediction System</h1>', unsafe_allow_html=True)
st.markdown("""
This application predicts palm oil prices using machine learning models. 
**Note:** If R² scores are negative, your models are worse than predicting the mean!
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
            df['Date'] = pd.date_range(start='2020-01-01', end='2025-05-31', periods=len(df))
        
        
        # Check if we have a Price column
        if 'Price' not in df.columns:
            st.error("⚠️ 'Price' column not found in dataset!")
            st.info("Please ensure your CSV file has a 'Price' column or rename your target column to 'Price'.")
            
            # Try to identify target column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.info(f"Available numeric columns: {list(numeric_cols)}")
                st.info("Please rename your target column to 'Price' in the CSV file.")
                return None
        
        # Check for missing values in Price
        if df['Price'].isnull().sum() > 0:
            st.warning(f"Found {df['Price'].isnull().sum()} missing values in Price column. Filling with median.")
            df['Price'] = df['Price'].fillna(df['Price'].median())
        
        # Check if Price column has variance
        if df['Price'].std() < 1e-10:
            st.error("⚠️ Price column has no variance (all values are almost identical)!")
            st.info("This will cause models to fail. Please check your data.")
            return None
        
        return df
    
    except FileNotFoundError:
        st.error("File 'price.csv' not found. Please create a CSV file with your data.")
        st.info("""
        Required columns:
        1. 'Date' - Date of observation
        2. 'Price' - Palm oil price (target variable)
        3. Other features: Solarradiation, Solarenergy, Uvindex, etc.
        """)
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def train_single_model_with_checks(X_train, y_train, X_test, y_test, model_name, tuned=True):
    """Train a single model with comprehensive checks"""
    
    # Check if we have enough data
    if len(X_train) < 10 or len(X_test) < 5:
        return None, None, None, "Not enough data for training/testing"
    
    # Check for variance in target
    if y_train.std() < 1e-10 or y_test.std() < 1e-10:
        return None, None, None, "Target variable has no variance"
    
    try:
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define model based on type
        if model_name == 'Random Forest':
            if tuned:
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [5, 10],
                    'min_samples_leaf': [1, 2]
                }
                model = GridSearchCV(
                    RandomForestRegressor(random_state=42),
                    param_grid,
                    cv=3,
                    scoring='r2',
                    n_jobs=-1
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
        
        elif model_name == 'XGBoost':
            if tuned:
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5],
                    'learning_rate': [0.01, 0.1]
                }
                model = GridSearchCV(
                    XGBRegressor(random_state=42),
                    param_grid,
                    cv=3,
                    scoring='r2',
                    n_jobs=-1
                )
            else:
                model = XGBRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
        
        elif model_name == 'Gradient Boosting':
            if tuned:
                param_grid = {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                }
                model = GridSearchCV(
                    GradientBoostingRegressor(random_state=42),
                    param_grid,
                    cv=3,
                    scoring='r2',
                    n_jobs=-1
                )
            else:
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                )
        
        elif model_name == 'SVR':
            # SVR is sensitive to scaling, so we'll use StandardScaler
            if tuned:
                param_grid = {
                    'C': [0.1, 1, 10],
                    'epsilon': [0.01, 0.1]
                }
                model = GridSearchCV(
                    SVR(kernel='rbf'),
                    param_grid,
                    cv=3,
                    scoring='r2',
                    n_jobs=-1
                )
            else:
                model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        
        elif model_name == 'Decision Tree':
            if tuned:
                param_grid = {
                    'max_depth': [3, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                model = GridSearchCV(
                    DecisionTreeRegressor(random_state=42),
                    param_grid,
                    cv=3,
                    scoring='r2',
                    n_jobs=-1
                )
            else:
                model = DecisionTreeRegressor(
                    max_depth=5,
                    min_samples_leaf=2,
                    random_state=42
                )
        
        else:
            return None, None, None, f"Unknown model: {model_name}"
        
        # Train the model
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics with checks
        try:
            r2 = r2_score(y_test, y_pred)
            
            # Check for absurd R² values
            if r2 < -1 or r2 > 1:
                return None, None, None, f"Invalid R² value: {r2}"
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            metrics = {
                'R²': r2,
                'RMSE': rmse,
                'MAE': mae
            }
            
            return model, metrics, scaler, training_time
            
        except Exception as e:
            return None, None, None, f"Error calculating metrics: {str(e)}"
            
    except Exception as e:
        return None, None, None, f"Error training model: {str(e)}"

@st.cache_resource
def train_all_models(X_train, y_train, X_test, y_test):
    """Train all models (tuned and untuned)"""
    models = {}
    metrics = {}
    scalers = {}
    training_times = {}
    errors = {}
    
    model_names = ['Random Forest', 'XGBoost', 'Gradient Boosting', 'SVR', 'Decision Tree']
    
    # Train tuned models
    for model_name in model_names:
        model_key = f"{model_name} (Tuned)"
        model, model_metrics, scaler, train_time = train_single_model_with_checks(
            X_train, y_train, X_test, y_test, model_name, tuned=True
        )
        
        if model is not None:
            models[model_key] = model
            metrics[model_key] = model_metrics
            scalers[model_key] = scaler
            training_times[model_key] = train_time
        else:
            errors[model_key] = model_metrics  # model_metrics contains error message
    
    # Train untuned models
    for model_name in model_names:
        model_key = f"{model_name} (Untuned)"
        model, model_metrics, scaler, train_time = train_single_model_with_checks(
            X_train, y_train, X_test, y_test, model_name, tuned=False
        )
        
        if model is not None:
            models[model_key] = model
            metrics[model_key] = model_metrics
            scalers[model_key] = scaler
            training_times[model_key] = train_time
        else:
            errors[model_key] = model_metrics
    
    return models, metrics, scalers, training_times, errors

# Load data
df = load_data()

if df is None:
    st.stop()

if page == "Data Overview":
    st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
    
    # Display critical warnings first
    if 'Price' in df.columns:
        price_stats = df['Price'].describe()
        
        # Check for data issues
        if price_stats['std'] < 1:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.error("⚠️ CRITICAL ISSUE: Price column has almost no variation!")
            st.write(f"Standard deviation: {price_stats['std']:.6f}")
            st.write("This will cause all models to fail. Please check your data.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if df['Price'].isnull().sum() > 0:
            st.warning(f"⚠️ Found {df['Price'].isnull().sum()} missing values in Price column")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Data Sample")
        st.dataframe(df.head(10), width='stretch')
    
    with col2:
        st.subheader("📊 Data Information")
        buffer = []
        buffer.append(f"**Total Rows:** {df.shape[0]}")
        buffer.append(f"**Total Columns:** {df.shape[1]}")
        if 'Date' in df.columns:
            buffer.append(f"**Date Range:** {df['Date'].min().date()} to {df['Date'].max().date()}")
        if 'Price' in df.columns:
            buffer.append(f"**Average Price:** $ {df['Price'].mean():.2f}")
            buffer.append(f"**Price Std Dev:** $ {df['Price'].std():.2f}")
            buffer.append(f"**Price Range:** ${df['Price'].min():.2f}-${df['Price'].max():.2f}")
        
        st.write("\n\n".join(buffer))
    
    # Line chart of price over time with yearly breakdown
    if 'Date' in df.columns and 'Price' in df.columns:
        st.subheader("📈 Yearly Price Trend Chart")
        
        # Create a copy of the dataframe for plotting
        plot_df = df.copy()
        
        # Ensure we have a Year column
        if 'Year' not in plot_df.columns:
            plot_df['Year'] = plot_df['Date'].dt.year
        
        # Create the line plot with yearly breakdown
        fig, ax = plt.subplots(figsize=(15, 7))
        
        # Get unique years in the data
        years = sorted(plot_df['Year'].unique())
        
        # Use viridis colormap
        colors = plt.cm.viridis(np.linspace(0, 1, len(years)))
        
        # Plot each year separately
        for year, color in zip(years, colors):
            year_data = plot_df[plot_df['Year'] == year]
            if len(year_data) > 0:
                ax.plot(year_data['Date'], year_data['Price'], 
                       label=str(year), color=color, linewidth=2)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_title('Palm Oil Price Trends by Year', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(title='Year', title_fontsize=12, fontsize=10)
        
        # Format x-axis dates
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Also show a boxplot of prices by year for comparison
        st.subheader("📊 Yearly Price Distribution")
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        # Create boxplot by year
        box_data = []
        box_labels = []
        for year in years:
            year_prices = plot_df[plot_df['Year'] == year]['Price'].dropna()
            if len(year_prices) > 0:
                box_data.append(year_prices)
                box_labels.append(str(year))
        
        if box_data:
            bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
            
            # Color the boxes with viridis colors
            for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Color the median lines
            for median in bp['medians']:
                median.set_color('red')
                median.set_linewidth(2)
            
            ax2.set_xlabel('Year', fontsize=12)
            ax2.set_ylabel('Price ($)', fontsize=12)
            ax2.set_title('Price Distribution by Year', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add mean markers
            for i, year_data in enumerate(box_data, 1):
                mean_price = year_data.mean()
                ax2.plot(i, mean_price, 'o', color='black', markersize=8)
                ax2.text(i, mean_price, f' ${mean_price:.2f}', 
                        ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig2)
    
    # Basic statistics
    st.subheader("📊 Statistical Summary")
    if 'Price' in df.columns:
        price_stats = df['Price'].describe()
        cols = st.columns(4)
        with cols[0]:
            st.metric("Mean", f"${price_stats['mean']:.2f}")
        with cols[1]:
            st.metric("Std Dev", f"${price_stats['std']:.2f}")
            if price_stats['std'] < 10:
                st.caption("⚠️ Low variance!")
        with cols[2]:
            st.metric("Min", f"${price_stats['min']:.2f}")
        with cols[3]:
            st.metric("Max", f"${price_stats['max']:.2f}")
    
    # Correlation analysis
    st.subheader("🔗 Correlation with Price")
    if 'Price' in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = {}
        
        for col in numeric_cols:
            if col != 'Price':
                corr = df[col].corr(df['Price'])
                correlations[col] = corr
        
        if correlations:
            corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation'])
            corr_df = corr_df.sort_values('Correlation', ascending=False)
            
            # Display top and bottom correlations
            st.write("**Top positive correlations:**")
            top_pos = corr_df[corr_df['Correlation'] > 0].head(5)
            if len(top_pos) > 0:
                st.dataframe(top_pos.style.format({'Correlation': '{:.3f}'}), width='stretch')
            else:
                st.write("No positive correlations found")
            
            st.write("**Top negative correlations:**")
            top_neg = corr_df[corr_df['Correlation'] < 0].head(5)
            if len(top_neg) > 0:
                st.dataframe(top_neg.style.format({'Correlation': '{:.3f}'}), width='stretch')
            else:
                st.write("No negative correlations found")

elif page == "Model Predictions":
    st.markdown('<h2 class="sub-header">Model Training & Prediction</h2>', unsafe_allow_html=True)
    
    if 'Price' not in df.columns:
        st.error("'Price' column not found in dataset. Cannot proceed with modeling.")
        st.stop()
    
    # Check data quality
    if df['Price'].std() < 1:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.error("🚨 CRITICAL DATA ISSUE")
        st.write(f"Price standard deviation: ${df['Price'].std():.6f}")
        st.write("Your Price column has almost no variation. Models will fail.")
        st.write("**Possible fixes:**")
        st.write("1. Check if your Price column has meaningful values")
        st.write("2. Ensure Price is not constant or near-constant")
        st.write("3. Check if Price column has the correct data type")
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()
    
    # Select features
    possible_features = ['Solarradiation', 'Solarenergy', 'Uvindex', 
                        'Index Production', 'Export Number (in Tonnes)', 
                        'USD', 'Year', 'Month', 'Day', 'DayOfYear']
    
    # Get available features
    available_features = [f for f in possible_features if f in df.columns]
    
    # Add other numeric features (excluding Price)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Price' in numeric_cols:
        numeric_cols.remove('Price')
    available_features = list(set(available_features + numeric_cols))
    
    if len(available_features) < 1:
        st.error("No features available for modeling!")
        st.info(f"Available columns: {df.columns.tolist()}")
        st.stop()
    
    # Prepare features and target
    X = df[available_features].copy()
    y = df['Price'].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    y = y.fillna(y.median())
    
    # Check for constant columns in features
    constant_cols = []
    for col in X.columns:
        if X[col].std() < 1e-10:
            constant_cols.append(col)
    
    if constant_cols:
        st.warning(f"Found constant columns: {constant_cols}. Removing them.")
        X = X.drop(columns=constant_cols)
        available_features = [f for f in available_features if f not in constant_cols]
    
    # Split data
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )
    
    # Display info
    with st.expander("📊 Data Information"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(X))
        with col2:
            st.metric("Training Samples", len(X_train))
        with col3:
            st.metric("Test Samples", len(X_test))
        with col4:
            st.metric("Features", len(available_features))
        
        st.write(f"**Target (Price) statistics:**")
        st.write(f"- Train mean: ${y_train.mean():.2f}, std: ${y_train.std():.2f}")
        st.write(f"- Test mean: ${y_test.mean():.2f}, std: ${y_test.std():.2f}")
        
        if y_train.std() < 1 or y_test.std() < 1:
            st.error("⚠️ Target variable has very low variance in train/test split!")
    
    # Feature correlation with target
    with st.expander("🔗 Feature Correlations with Price"):
        train_corrs = {}
        for col in X.columns:
            corr = np.corrcoef(X_train[col], y_train)[0, 1]
            train_corrs[col] = corr
        
        corr_df = pd.DataFrame.from_dict(train_corrs, orient='index', columns=['Correlation'])
        corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
        
        # Highlight strong correlations
        def highlight_correlation(val):
            if abs(val) > 0.7:
                return 'background-color: #4CAF50; color: white'
            elif abs(val) > 0.5:
                return 'background-color: #8BC34A'
            elif abs(val) > 0.3:
                return 'background-color: #CDDC39'
            elif abs(val) < 0.1:
                return 'background-color: #FF9800'
            else:
                return ''
        
        st.dataframe(
            corr_df.style.format({'Correlation': '{:.3f}'})
            .applymap(highlight_correlation, subset=['Correlation']),
            width='stretch'
        )
        
        # Warning if no strong correlations
        strong_corrs = corr_df[np.abs(corr_df['Correlation']) > 0.3]
        if len(strong_corrs) == 0:
            st.warning("⚠️ No features have strong correlation (>0.3) with Price!")
            st.write("This might explain poor model performance.")
    
    # Train models button
    if st.button("🚀 Train All Models", type="primary"):
        with st.spinner("Training models... This may take a minute."):
            # Clear previous session state
            for key in ['all_models', 'all_metrics', 'all_scalers', 'training_times', 'training_errors']:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Train all models
            models, metrics, scalers, training_times, errors = train_all_models(
                X_train, y_train, X_test, y_test
            )
            
            # Store in session state
            st.session_state.all_models = models
            st.session_state.all_metrics = metrics
            st.session_state.all_scalers = scalers
            st.session_state.training_times = training_times
            st.session_state.training_errors = errors
            st.session_state.model_trained = True
        
        # Display results
        if errors:
            st.warning("⚠️ Some models failed to train:")
            for model_name, error_msg in errors.items():
                st.write(f"- **{model_name}:** {error_msg}")
        
        if metrics:
            st.success(f"✅ Successfully trained {len(metrics)} models!")
    
    # Display results if models are trained
    if 'model_trained' in st.session_state and st.session_state.model_trained:
        if 'all_metrics' not in st.session_state or not st.session_state.all_metrics:
            st.warning("No models were successfully trained.")
            st.stop()
        
        all_metrics = st.session_state.all_metrics
        
        # Display all model performances
        st.subheader("📊 Model Performance Summary")
        
        # Create performance table
        perf_data = []
        for model_name, model_metrics in all_metrics.items():
            perf_data.append({
                'Model': model_name,
                'R²': model_metrics['R²'],
                'RMSE': model_metrics['RMSE'],
                'MAE': model_metrics['MAE']
            })
        
        perf_df = pd.DataFrame(perf_data)
        perf_df = perf_df.sort_values('R²', ascending=False)
        
        # Highlight negative R²
        def highlight_r2(val):
            if val < 0:
                return 'background-color: #ff4444; color: white; font-weight: bold'
            elif val < 0.3:
                return 'background-color: #ff9800'
            elif val < 0.6:
                return 'background-color: #ffeb3b'
            elif val < 0.8:
                return 'background-color: #8bc34a'
            else:
                return 'background-color: #4CAF50; color: white'
        
        # Display table
        st.dataframe(
            perf_df.style.format({
                'R²': '{:.4f}',
                'RMSE': '{:.2f}',
                'MAE': '{:.2f}'
            }).applymap(highlight_r2, subset=['R²']),
            width='stretch'
        )
        
        # Warning about negative R²
        negative_r2_models = perf_df[perf_df['R²'] < 0]['Model'].tolist()
        if negative_r2_models:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.error(f"🚨 {len(negative_r2_models)} models have NEGATIVE R²!")
            st.write("These models perform WORSE than just predicting the mean of Price.")
            st.write("**Models with negative R²:** " + ", ".join(negative_r2_models))
            st.write("**Possible causes:**")
            st.write("1. Features have no predictive power")
            st.write("2. Data leakage issues")
            st.write("3. Incorrect model configuration")
            st.write("4. Overfitting to noise")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Select model for detailed view
        st.subheader("🔍 Detailed Model Analysis")
        
        selected_model = st.selectbox(
            "Choose a model for detailed analysis:",
            list(all_metrics.keys()),
            index=0
        )
        
        if selected_model in st.session_state.all_models:
            model = st.session_state.all_models[selected_model]
            metrics = st.session_state.all_metrics[selected_model]
            scaler = st.session_state.all_scalers[selected_model]
            
            # Make predictions
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                color = "red" if metrics['R²'] < 0 else "green"
                st.metric("R² Score", f"{metrics['R²']:.4f}", delta=None, delta_color="off")
                if metrics['R²'] < 0:
                    st.caption("⚠️ Worse than predicting mean!")
                elif metrics['R²'] < 0.3:
                    st.caption("Poor predictive power")
                elif metrics['R²'] < 0.6:
                    st.caption("Moderate predictive power")
                elif metrics['R²'] < 0.8:
                    st.caption("Good predictive power")
                else:
                    st.caption("Excellent predictive power")
            
            with col2:
                st.metric("RMSE", f"{metrics['RMSE']:.2f}")
                st.caption(f"Average error: ${metrics['RMSE']:.2f}")
            
            with col3:
                st.metric("MAE", f"{metrics['MAE']:.2f}")
                st.caption(f"Average absolute error: ${metrics['MAE']:.2f}")
            
            # Visualizations
            st.subheader("📈 Prediction Visualizations")
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. Actual vs Predicted scatter
            ax1.scatter(y_test, y_pred, alpha=0.5, color='#2E8B57')
            ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                    'r--', label='Perfect Prediction')
            ax1.set_xlabel('Actual Price ($)')
            ax1.set_ylabel('Predicted Price ($)')
            ax1.set_title(f'{selected_model}: Actual vs Predicted')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Residuals plot
            residuals = y_test - y_pred
            ax2.scatter(y_pred, residuals, alpha=0.5, color='#FF6B6B')
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_xlabel('Predicted Price ($)')
            ax2.set_ylabel('Residuals (Actual - Predicted)')
            ax2.set_title(f'Residual Plot')
            ax2.grid(True, alpha=0.3)
            
            # 3. Distribution of residuals
            ax3.hist(residuals, bins=30, color='#2196F3', edgecolor='black', alpha=0.7)
            ax3.axvline(x=0, color='r', linestyle='--')
            ax3.set_xlabel('Residuals')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of Residuals')
            ax3.grid(True, alpha=0.3)
            
            # 4. Time series of predictions (if dates available in test)
            if 'Date' in df.columns:
                # Get dates for test set
                test_dates = df['Date'].iloc[-len(y_test):].reset_index(drop=True)
                ax4.plot(test_dates, y_test.values, label='Actual', color='#2E8B57', linewidth=2)
                ax4.plot(test_dates, y_pred, label='Predicted', color='#FF9800', linewidth=2, alpha=0.7)
                ax4.set_xlabel('Date')
                ax4.set_ylabel('Price ($)')
                ax4.set_title('Time Series: Actual vs Predicted')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
            else:
                # If no dates, show error distribution
                error_pct = abs(residuals / y_test * 100)
                ax4.hist(error_pct, bins=30, color='#9C27B0', edgecolor='black', alpha=0.7)
                ax4.set_xlabel('Error Percentage (%)')
                ax4.set_ylabel('Frequency')
                ax4.set_title('Distribution of Percentage Errors')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show predictions table
            st.subheader("📋 Sample Predictions")
            
            results_df = pd.DataFrame({
                'Actual': y_test.values[:20],
                'Predicted': y_pred[:20],
                'Error': y_pred[:20] - y_test.values[:20],
                'Error %': abs((y_pred[:20] - y_test.values[:20]) / y_test.values[:20] * 100)
            })
            
            st.dataframe(
                results_df.style.format({
                    'Actual': '${:.2f}',
                    'Predicted': '${:.2f}',
                    'Error': '${:.2f}',
                    'Error %': '{:.1f}%'
                }),
                width='stretch'
            )

elif page == "Results Comparison":
    st.markdown('<h2 class="sub-header">Model Performance Comparison</h2>', unsafe_allow_html=True)
    
    if 'model_trained' not in st.session_state or not st.session_state.model_trained:
        st.warning("⚠️ No models have been trained yet!")
        st.info("Please go to 'Model Predictions' page and train models first.")
        st.stop()
    
    if 'all_metrics' not in st.session_state or not st.session_state.all_metrics:
        st.warning("No model metrics available. Please train models first.")
        st.stop()
    
    all_metrics = st.session_state.all_metrics
    
    # Create comparison table
    st.subheader("📊 Tuned vs Untuned Performance Comparison")
    
    comparison_data = []
    for model_name, metrics in all_metrics.items():
        comparison_data.append({
            'Model': model_name,
            'Type': 'Tuned' if '(Tuned)' in model_name else 'Untuned',
            'Base Model': model_name.replace(' (Tuned)', '').replace(' (Untuned)', ''),
            'R²': metrics['R²'],
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE']
        })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Sort by R²
    comp_df = comp_df.sort_values('R²', ascending=False).reset_index(drop=True)
    
    # Display comparison
    st.dataframe(
        comp_df.style.format({
            'R²': '{:.4f}',
            'RMSE': '{:.2f}',
            'MAE': '{:.2f}'
        }).apply(lambda x: ['background-color: #e8f5e9' if v == 'Tuned' else 
                           'background-color: #fff3e0' for v in x], 
                subset=['Type']),
        width='stretch'
    )
    
    # Analyze tuned vs untuned performance
    st.subheader("📈 Tuning Effectiveness Analysis")
    
    # Group by base model
    base_models = comp_df['Base Model'].unique()
    
    improvement_data = []
    for base_model in base_models:
        tuned_row = comp_df[(comp_df['Base Model'] == base_model) & (comp_df['Type'] == 'Tuned')]
        untuned_row = comp_df[(comp_df['Base Model'] == base_model) & (comp_df['Type'] == 'Untuned')]
        
        if len(tuned_row) > 0 and len(untuned_row) > 0:
            tuned_r2 = tuned_row.iloc[0]['R²']
            untuned_r2 = untuned_row.iloc[0]['R²']
            r2_improvement = tuned_r2 - untuned_r2
            
            tuned_rmse = tuned_row.iloc[0]['RMSE']
            untuned_rmse = untuned_row.iloc[0]['RMSE']
            rmse_improvement = untuned_rmse - tuned_rmse  # Positive is improvement
            
            improvement_data.append({
                'Model': base_model,
                'Tuned R²': tuned_r2,
                'Untuned R²': untuned_r2,
                'R² Improvement': r2_improvement,
                'Tuned RMSE': tuned_rmse,
                'Untuned RMSE': untuned_rmse,
                'RMSE Improvement': rmse_improvement
            })
    
    if improvement_data:
        improvement_df = pd.DataFrame(improvement_data)
        
        # Display improvements
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**R² Improvement from Tuning:**")
            improvement_df = improvement_df.sort_values('R² Improvement', ascending=False)
            for _, row in improvement_df.iterrows():
                if row['R² Improvement'] > 0:
                    st.success(f"✅ {row['Model']}: +{row['R² Improvement']:.4f}")
                else:
                    st.error(f"❌ {row['Model']}: {row['R² Improvement']:.4f}")
        
        with col2:
            st.write("**RMSE Improvement from Tuning:**")
            improvement_df = improvement_df.sort_values('RMSE Improvement', ascending=False)
            for _, row in improvement_df.iterrows():
                if row['RMSE Improvement'] > 0:
                    st.success(f"✅ {row['Model']}: -{row['RMSE Improvement']:.2f}")
                else:
                    st.error(f"❌ {row['Model']}: +{abs(row['RMSE Improvement']):.2f}")
    
    # Best model recommendation
    st.subheader("🎯 Best Performing Model")
    
    best_row = comp_df.iloc[0]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div style='background-color: #FFF8DC; padding: 20px; border-radius: 10px; border: 3px solid #FFD700;'>
            <h3>🥇 {best_row['Model']}</h3>
            <p><strong>R²:</strong> {best_row['R²']:.4f}</p>
            <p><strong>RMSE:</strong> ${best_row['RMSE']:.2f}</p>
            <p><strong>MAE:</strong> ${best_row['MAE']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.write("**Why this model is recommended:**")
        
        if best_row['R²'] < 0:
            st.error("⚠️ Even the best model has negative R²!")
            st.write("This indicates serious data or modeling issues.")
        elif best_row['R²'] < 0.3:
            st.warning("⚠️ Best model has poor predictive power")
            st.write("Consider improving your features or data quality.")
        elif best_row['R²'] < 0.6:
            st.info("📊 Model has moderate predictive power")
            st.write("Could be useful for some applications.")
        elif best_row['R²'] < 0.8:
            st.success("✅ Good predictive power")
            st.write("Suitable for most applications.")
        else:
            st.success("🎯 Excellent predictive power!")
            st.write("Very reliable for predictions.")
        
        if "(Tuned)" in best_row['Model']:
            st.info("🔧 This is a tuned model with optimized hyperparameters")
        else:
            st.info("⚙️ This is an untuned model with default parameters")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Developed using Streamlit | BSD3523 Machine Learning Project</p>
    <p>Group: CSM1 | University Malaysia Pahang Al-Sultan Abdullah</p>
    <p style='font-size: 0.9em; color: #666;'>
        Note: Negative R² indicates models worse than predicting the mean
    </p>
</div>
""", unsafe_allow_html=True)
