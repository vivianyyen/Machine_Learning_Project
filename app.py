# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
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
Using features from uploaded dataset only.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Data Overview", "Model Predictions", "Results Comparison"])

# Cache data loading
@st.cache_data
def load_data():
    """Load and prepare data from price.csv"""
    try:
        # Try to load data
        df = pd.read_csv("price.csv")
        
        # Log data loading
        st.sidebar.info(f"📊 Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Show actual columns in dataset
        st.sidebar.write("**📋 Actual columns in price.csv:**")
        for col in df.columns:
            st.sidebar.write(f"- {col}")
        
        # Ensure Date column is datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        elif 'date' in df.columns:
            df['Date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.drop('date', axis=1)
        else:
            # If no Date column, check for any date-like column
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                df['Date'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
                df = df.drop(date_cols[0], axis=1)
            else:
                # Create a simple date range
                df['Date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
        
        # Check for Price column (case insensitive)
    price_col = None
    for col in df.columns:
        if 'price' in col.lower():
            price_col = col
            break
    
    if price_col and price_col != 'Price':
        df['Price'] = df[price_col]
        df = df.drop(price_col, axis=1)
        st.sidebar.success(f"✓ Found price column: '{price_col}' renamed to 'Price'")
    
    if 'Price' not in df.columns:
        st.error("❌ 'Price' column not found in dataset!")
        st.write("**Available columns in your dataset:**", df.columns.tolist())
        st.write("Please ensure your CSV has a column named 'Price' or containing 'price' in its name.")
        return None
    
    # Fill missing values in Price
    if df['Price'].isnull().sum() > 0:
        st.warning(f"⚠️ Found {df['Price'].isnull().sum()} missing values in Price column. Filling with median.")
        df['Price'] = df['Price'].fillna(df['Price'].median())
    
    # Check Price variance
    price_std = df['Price'].std()
    if price_std < 1e-10:
        st.error(f"❌ Price column has no variance (std: {price_std:.6f})!")
        st.write("All models will fail with negative R² if price doesn't change.")
        return None
    
    # Fill other numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    return df

except FileNotFoundError:
    st.error("❌ File 'price.csv' not found!")
    st.info("""
    Please ensure you have a file named 'price.csv' in the same directory as this app.
    Your file should contain:
    1. A 'Price' column (palm oil prices)
    2. A 'Date' column (optional)
    3. Other feature columns
    """)
    return None
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    return None

@st.cache_resource
def train_models(X_train, y_train, X_test, y_test, model_type='tuned'):
    """Train models with the given data"""
    
    models = {}
    metrics = {}
    scalers = {}
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model configurations
    if model_type == 'tuned':
        model_configs = {
            'Random Forest': GridSearchCV(
                RandomForestRegressor(random_state=42, n_jobs=-1),
                {'n_estimators': [50, 100], 'max_depth': [5, 10]},
                cv=3, scoring='r2', n_jobs=-1
            ),
            'XGBoost': GridSearchCV(
                XGBRegressor(random_state=42, n_jobs=-1),
                {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]},
                cv=3, scoring='r2', n_jobs=-1
            ),
            'Gradient Boosting': GridSearchCV(
                GradientBoostingRegressor(random_state=42),
                {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},
                cv=3, scoring='r2'
            ),
            'SVR': GridSearchCV(
                SVR(),
                {'C': [1, 10], 'epsilon': [0.1, 0.2]},
                cv=3, scoring='r2'
            ),
            'Decision Tree': GridSearchCV(
                DecisionTreeRegressor(random_state=42),
                {'max_depth': [3, 5, 10], 'min_samples_leaf': [1, 2, 4]},
                cv=3, scoring='r2'
            ),
            'Linear Regression': Ridge(alpha=1.0, random_state=42)  # Simple Ridge for stability
        }
    else:  # untuned
        model_configs = {
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'XGBoost': XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42),
            'SVR': SVR(C=1.0, epsilon=0.1),
            'Decision Tree': DecisionTreeRegressor(max_depth=5, min_samples_leaf=2, random_state=42),
            'Linear Regression': Ridge(alpha=1.0, random_state=42)
        }
    
    # Train each model
    for name, model in model_configs.items():
        try:
            # Train model
            start_time = time.time()
            model.fit(X_train_scaled, y_train)
            train_time = time.time() - start_time
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics[name] = {
                'R²': r2_score(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred),
                'Time': train_time
            }
            
            models[name] = model
            scalers[name] = scaler
            
        except Exception as e:
            st.warning(f"Could not train {name}: {str(e)}")
    
    return models, metrics, scalers

# Load data
df = load_data()

if df is None:
    st.stop()

if page == "Data Overview":
    st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
    
    # Show data statistics
    st.write(f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 First 10 Rows")
        st.dataframe(df.head(10), width='stretch')
    
    with col2:
        st.subheader("📊 Data Information")
        
        info_text = []
        info_text.append(f"**Total Rows:** {df.shape[0]}")
        info_text.append(f"**Total Columns:** {df.shape[1]}")
        
        if 'Date' in df.columns:
            info_text.append(f"**Date Range:** {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        if 'Price' in df.columns:
            price_stats = df['Price'].describe()
            info_text.append(f"**Price Statistics:**")
            info_text.append(f"- Mean: ${price_stats['mean']:.2f}")
            info_text.append(f"- Std Dev: ${price_stats['std']:.2f}")
            info_text.append(f"- Min: ${price_stats['min']:.2f}")
            info_text.append(f"- Max: ${price_stats['max']:.2f}")
            
            # Check for low variance
            if price_stats['std'] < 1:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.error("⚠️ Price has very low variation!")
                st.write(f"Standard deviation: ${price_stats['std']:.6f}")
                st.write("This will cause negative R² scores.")
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.write("\n".join(info_text))
    
    # FIXED LINE CHART - Simple and clear
    if 'Date' in df.columns and 'Price' in df.columns:
        st.subheader("📈 Price Trend Over Time")
        
        # Create a clean figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot price vs date
        ax.plot(df['Date'], df['Price'], color='#2E8B57', linewidth=2)
        
        # Set labels and title
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_title('Palm Oil Price Trend', fontsize=14, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(fig)
        
        # Also show a simple scatter plot of recent prices
        st.subheader("📊 Recent Price Distribution")
        
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        
        # Show last 100 prices or all if less
        n_points = min(100, len(df))
        recent_prices = df['Price'].tail(n_points)
        
        ax2.hist(recent_prices, bins=20, color='#3CB371', edgecolor='black', alpha=0.7)
        ax2.axvline(x=recent_prices.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${recent_prices.mean():.2f}')
        
        ax2.set_xlabel('Price ($)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'Distribution of Last {n_points} Prices', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig2)
    
    # Show all features in dataset
    st.subheader("🔍 All Features in Dataset")
    
    feature_info = []
    for col in df.columns:
        if col not in ['Date', 'Price']:
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            
            if np.issubdtype(df[col].dtype, np.number):
                # Numeric feature
                mean_val = df[col].mean()
                std_val = df[col].std()
                feature_info.append({
                    'Feature': col,
                    'Type': dtype,
                    'Null': null_count,
                    'Unique': unique_count,
                    'Mean': f"{mean_val:.2f}",
                    'Std': f"{std_val:.2f}"
                })
            else:
                # Non-numeric feature
                feature_info.append({
                    'Feature': col,
                    'Type': dtype,
                    'Null': null_count,
                    'Unique': unique_count,
                    'Mean': 'N/A',
                    'Std': 'N/A'
                })
    
    if feature_info:
        feature_df = pd.DataFrame(feature_info)
        st.dataframe(feature_df, width='stretch')
    else:
        st.info("No additional features found besides Date and Price.")

elif page == "Model Predictions":
    st.markdown('<h2 class="sub-header">Model Training & Prediction</h2>', unsafe_allow_html=True)
    
    if 'Price' not in df.columns:
        st.error("'Price' column not found. Cannot proceed.")
        st.stop()
    
    # Get all numeric columns except Price as features
    all_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove Price from features
    if 'Price' in all_features:
        all_features.remove('Price')
    
    if not all_features:
        st.error("❌ No numeric features found in dataset!")
        st.write("Please ensure your CSV has numeric columns (besides Price) for prediction.")
        st.stop()
    
    # Let user select which features to use
    st.sidebar.subheader("Feature Selection")
    st.write(f"**Available numeric features:** {len(all_features)} found")
    
    # Show feature correlations with Price
    correlations = {}
    for feature in all_features:
        corr = df[feature].corr(df['Price'])
        correlations[feature] = corr
    
    # Sort features by absolute correlation
    sorted_features = sorted(all_features, key=lambda x: abs(correlations.get(x, 0)), reverse=True)
    
    # Select features (default: top 5 by correlation)
    selected_features = st.sidebar.multiselect(
        "Select features to use:",
        sorted_features,
        default=sorted_features[:min(5, len(sorted_features))]
    )
    
    if not selected_features:
        st.warning("Please select at least one feature.")
        st.stop()
    
    # Prepare data
    X = df[selected_features].copy()
    y = df['Price'].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    y = y.fillna(y.median())
    
    # Split data
    test_size = st.sidebar.slider("Test size (%)", 10, 40, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Show data info
    with st.expander("📊 Data Information"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(X))
        with col2:
            st.metric("Training Samples", len(X_train))
        with col3:
            st.metric("Test Samples", len(X_test))
        with col4:
            st.metric("Features", len(selected_features))
        
        st.write("**Selected Features:**")
        for feature in selected_features:
            corr = correlations.get(feature, 0)
            if corr > 0.5:
                st.success(f"✓ {feature}: corr = {corr:.3f}")
            elif corr > 0.3:
                st.info(f"• {feature}: corr = {corr:.3f}")
            elif corr > 0:
                st.write(f"- {feature}: corr = {corr:.3f}")
            elif corr < -0.3:
                st.warning(f"⚠ {feature}: corr = {corr:.3f}")
            else:
                st.write(f"- {feature}: corr = {corr:.3f}")
    
    # Model selection
    st.sidebar.subheader("Model Selection")
    train_tuned = st.sidebar.checkbox("Train Tuned Models", value=True)
    train_untuned = st.sidebar.checkbox("Train Untuned Models", value=True)
    
    model_options = {
        'Linear Regression': st.sidebar.checkbox("Linear Regression", value=True),
        'Random Forest': st.sidebar.checkbox("Random Forest", value=True),
        'XGBoost': st.sidebar.checkbox("XGBoost", value=True),
        'Gradient Boosting': st.sidebar.checkbox("Gradient Boosting", value=True),
        'SVR': st.sidebar.checkbox("SVR", value=False),
        'Decision Tree': st.sidebar.checkbox("Decision Tree", value=True)
    }
    
    selected_models = [model for model, selected in model_options.items() if selected]
    
    if not selected_models:
        st.warning("Please select at least one model.")
        st.stop()
    
    # Train button
    if st.button("🚀 Train Selected Models", type="primary"):
        if not train_tuned and not train_untuned:
            st.warning("Please select at least one model type.")
            st.stop()
        
        with st.spinner("Training models..."):
            # Clear previous results
            for key in ['tuned_models', 'tuned_metrics', 'untuned_models', 'untuned_metrics']:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Train models
            if train_tuned:
                tuned_models, tuned_metrics, _ = train_models(
                    X_train, y_train, X_test, y_test, model_type='tuned'
                )
                # Filter to selected models only
                tuned_metrics = {k: v for k, v in tuned_metrics.items() if k in selected_models}
                st.session_state.tuned_models = tuned_models
                st.session_state.tuned_metrics = tuned_metrics
            
            if train_untuned:
                untuned_models, untuned_metrics, _ = train_models(
                    X_train, y_train, X_test, y_test, model_type='untuned'
                )
                # Filter to selected models only
                untuned_metrics = {k: v for k, v in untuned_metrics.items() if k in selected_models}
                st.session_state.untuned_models = untuned_models
                st.session_state.untuned_metrics = untuned_metrics
            
            st.session_state.model_trained = True
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.selected_features = selected_features
        
        st.success("✅ Training completed!")
    
    # Show results if trained
    if 'model_trained' in st.session_state and st.session_state.model_trained:
        # Combine all results
        all_results = {}
        
        if 'tuned_metrics' in st.session_state:
            for model, metrics in st.session_state.tuned_metrics.items():
                all_results[f"{model} (Tuned)"] = metrics
        
        if 'untuned_metrics' in st.session_state:
            for model, metrics in st.session_state.untuned_metrics.items():
                all_results[f"{model} (Untuned)"] = metrics
        
        if not all_results:
            st.warning("No results available.")
            st.stop()
        
        # Display results
        st.subheader("📊 Model Performance")
        
        results_data = []
        for model_name, metrics in all_results.items():
            results_data.append({
                'Model': model_name,
                'R²': metrics['R²'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'Time (s)': metrics['Time']
            })
        
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('R²', ascending=False)
        
        # Color code R²
        def color_r2(val):
            if val < 0:
                return 'background-color: #ff4444; color: white'
            elif val < 0.3:
                return 'background-color: #ff9800'
            elif val < 0.6:
                return 'background-color: #ffeb3b'
            elif val < 0.8:
                return 'background-color: #8bc34a'
            else:
                return 'background-color: #4CAF50; color: white'
        
        st.dataframe(
            results_df.style.format({
                'R²': '{:.4f}',
                'RMSE': '{:.2f}',
                'MAE': '{:.2f}',
                'Time (s)': '{:.2f}'
            }).applymap(color_r2, subset=['R²']),
            width='stretch'
        )
        
        # Warning for negative R²
        negative_count = results_df[results_df['R²'] < 0].shape[0]
        if negative_count > 0:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.warning(f"⚠️ {negative_count} models have negative R²!")
            st.write("This means they perform worse than just predicting the average price.")
            st.write("**Possible reasons:**")
            st.write("1. Features don't correlate with Price")
            st.write("2. Price has very low variation")
            st.write("3. Not enough training data")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Select model for detailed view
        st.subheader("🔍 Detailed Analysis")
        
        selected_result = st.selectbox(
            "Select a model for detailed view:",
            list(all_results.keys())
        )
        
        # Get the model
        if "(Tuned)" in selected_result:
            model_name = selected_result.replace(" (Tuned)", "")
            if 'tuned_models' in st.session_state and model_name in st.session_state.tuned_models:
                model = st.session_state.tuned_models[model_name]
                metrics = st.session_state.tuned_metrics[model_name]
        else:
            model_name = selected_result.replace(" (Untuned)", "")
            if 'untuned_models' in st.session_state and model_name in st.session_state.untuned_models:
                model = st.session_state.untuned_models[model_name]
                metrics = st.session_state.untuned_metrics[model_name]
        
        # Show metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{metrics['R²']:.4f}")
        with col2:
            st.metric("RMSE", f"{metrics['RMSE']:.2f}")
        with col3:
            st.metric("MAE", f"{metrics['MAE']:.2f}")
        
        # Make predictions for visualization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model.fit(X_train_scaled, y_train)  # Re-fit for consistency
        y_pred = model.predict(X_test_scaled)
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        ax1.scatter(y_test, y_pred, alpha=0.6, color='#2E8B57')
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', linewidth=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Price')
        ax1.set_ylabel('Predicted Price')
        ax1.set_title(f'{selected_result}: Actual vs Predicted')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Residuals
        residuals = y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6, color='#FF6B6B')
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted Price')
        ax2.set_ylabel('Residuals (Actual - Predicted)')
        ax2.set_title('Residual Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

elif page == "Results Comparison":
    st.markdown('<h2 class="sub-header">Results Comparison</h2>', unsafe_allow_html=True)
    
    if 'model_trained' not in st.session_state:
        st.warning("No models trained yet. Go to Model Predictions page first.")
        st.stop()
    
    # Combine results
    all_results = {}
    
    if 'tuned_metrics' in st.session_state:
        for model, metrics in st.session_state.tuned_metrics.items():
            all_results[f"{model} (Tuned)"] = metrics
    
    if 'untuned_metrics' in st.session_state:
        for model, metrics in st.session_state.untuned_metrics.items():
            all_results[f"{model} (Untuned)"] = metrics
    
    if not all_results:
        st.warning("No results available for comparison.")
        st.stop()
    
    # Create comparison table
    comparison_data = []
    for model_name, metrics in all_results.items():
        comparison_data.append({
            'Model': model_name,
            'Type': 'Tuned' if '(Tuned)' in model_name else 'Untuned',
            'Base Model': model_name.replace(' (Tuned)', '').replace(' (Untuned)', ''),
            'R²': metrics['R²'],
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE']
        })
    
    comp_df = pd.DataFrame(comparison_data)
    comp_df = comp_df.sort_values('R²', ascending=False)
    
    # Display
    def highlight_row(row):
        if row['Type'] == 'Tuned':
            return ['background-color: #e8f5e9'] * len(row)
        else:
            return ['background-color: #fff3e0'] * len(row)
    
    st.dataframe(
        comp_df.style.format({
            'R²': '{:.4f}',
            'RMSE': '{:.2f}',
            'MAE': '{:.2f}'
        }).apply(highlight_row, axis=1),
        width='stretch'
    )
    
    # Best model
    best_model = comp_df.iloc[0]
    st.subheader("🎯 Best Model")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div style='background-color: #FFF8DC; padding: 20px; border-radius: 10px; border: 2px solid #FFD700;'>
            <h3 style='margin-top: 0;'>🥇 {best_model['Model']}</h3>
            <p><strong>R²:</strong> {best_model['R²']:.4f}</p>
            <p><strong>RMSE:</strong> {best_model['RMSE']:.2f}</p>
            <p><strong>MAE:</strong> {best_model['MAE']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.write("**Performance Analysis:**")
        if best_model['R²'] < 0:
            st.error("⚠️ Even the best model has negative R²!")
            st.write("This indicates serious issues with your data or features.")
        elif best_model['R²'] < 0.3:
            st.warning("Poor predictive power")
        elif best_model['R²'] < 0.6:
            st.info("Moderate predictive power")
        elif best_model['R²'] < 0.8:
            st.success("Good predictive power")
        else:
            st.success("Excellent predictive power!")
        
        if "(Tuned)" in best_model['Model']:
            st.info("This is a tuned model with optimized parameters.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Developed using Streamlit | BSD3523 Machine Learning Project</p>
    <p>Group: CSM1 | University Malaysia Pahang Al-Sultan Abdullah</p>
</div>
""", unsafe_allow_html=True)
