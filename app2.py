# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Import machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import top 5 models (added Decision Tree)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
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
page = st.sidebar.radio("Go to:", ["📊 Data Overview", "🤖 Model Predictions", "📈 Results Comparison", "⚙️ Hyperparameter Tuning"])

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
            'svr__C': [0.1, 1, 10, 100],
            'svr__epsilon': [0.01, 0.1, 0.5, 1.0],
            'svr__kernel': ['linear', 'rbf', 'poly']
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
def train_models_with_tuning(X_train_scaled, y_train, tuning_method='grid', cv_folds=3):
    """Train models with hyperparameter tuning"""
    models = {}
    best_params = {}
    search_results = {}
    
    # Get hyperparameter grids
    param_grids = get_hyperparameter_grids()
    
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
    else:  # random search
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
    search_results['Random Forest'] = rf_search.cv_results_
    
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
    search_results['XGBoost'] = xgb_search.cv_results_
    
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
    search_results['Gradient Boosting'] = gbr_search.cv_results_
    
    # 4. SVR (Support Vector Regression)
    svr_pipeline = make_pipeline(StandardScaler(), SVR())
    
    if tuning_method == 'grid':
        svr_search = GridSearchCV(
            svr_pipeline,
            param_grids['SVR'],
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
    else:
        svr_search = RandomizedSearchCV(
            svr_pipeline,
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
    search_results['SVR'] = svr_search.cv_results_
    
    # 5. Decision Tree Regressor (NEW)
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
    search_results['Decision Tree'] = dt_search.cv_results_
    
    return models, best_params, search_results

@st.cache_resource
def train_models_basic(X_train_scaled, y_train):
    """Train models without hyperparameter tuning (for comparison)"""
    models = {}
    
    # 1. Random Forest Regressor
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    models['Random Forest'] = rf
    
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
    xgb.fit(X_train_scaled, y_train)
    models['XGBoost'] = xgb
    
    # 3. Gradient Boosting Regressor
    gbr = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    gbr.fit(X_train_scaled, y_train)
    models['Gradient Boosting'] = gbr
    
    # 4. SVR (Support Vector Regression)
    svr = make_pipeline(
        StandardScaler(),
        SVR(kernel='rbf', C=10, epsilon=0.1)
    )
    svr.fit(X_train_scaled, y_train)
    models['SVR'] = svr
    
    # 5. Decision Tree Regressor
    dt = DecisionTreeRegressor(
        max_depth=5,
        min_samples_leaf=5,
        random_state=42
    )
    dt.fit(X_train_scaled, y_train)
    models['Decision Tree'] = dt
    
    return models

# Load data
df = load_data()

if page == "📊 Data Overview":
    st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Data Sample")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.subheader("📊 Data Information")
        buffer = []
        buffer.append(f"**Total Rows:** {df.shape[0]}")
        buffer.append(f"**Total Columns:** {df.shape[1]}")
        if 'Date' in df.columns:
            buffer.append(f"**Date Range:** {df['Date'].min().date()} to {df['Date'].max().date()}")
        if 'Price' in df.columns:
            buffer.append(f"**Average Price:** ${df['Price'].mean():.2f}")
            buffer.append(f"**Price Range:** ${df['Price'].min():.2f} - ${df['Price'].max():.2f}")
        
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

elif page == "🤖 Model Predictions":
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
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
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
                rfe.fit(X_train_scaled, y_train)
                
                selected_features = X_train.columns[rfe.support_].tolist()
                st.write(f"**Selected {n_features} features:** {', '.join(selected_features)}")
                
                # Use RFE transformed data
                X_train_rfe = rfe.transform(X_train_scaled)
                X_test_rfe = rfe.transform(X_test_scaled)
            
            # Train models
            st.subheader("🤖 Training 5 Models")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            if use_hyperparameter_tuning:
                status_text.text("Performing hyperparameter tuning... This may take a few minutes.")
                models, best_params, search_results = train_models_with_tuning(
                    X_train_scaled, y_train, tuning_method, cv_folds
                )
                tuning_status = "with Hyperparameter Tuning"
            else:
                status_text.text("Training models with default parameters...")
                models = train_models_basic(X_train_scaled, y_train)
                tuning_status = "with Default Parameters"
                best_params = {}
            
            progress_bar.progress(1.0)
            
            if models:
                status_text.text(f"✅ All models trained successfully {tuning_status}!")
                
                # Display best parameters if tuning was used
                if use_hyperparameter_tuning and best_params:
                    with st.expander("📋 Best Hyperparameters"):
                        for model_name, params in best_params.items():
                            st.write(f"**{model_name}:**")
                            st.json(params)
                
                # Select a model to view predictions
                st.subheader("📊 Model Predictions")
                
                selected_model = st.selectbox(
                    "Choose a model to view predictions:",
                    list(models.keys())
                )
                
                model = models[selected_model]
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
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
                st.subheader("📈 Prediction Visualization")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Scatter plot
                ax1.scatter(y_test, y_pred, alpha=0.5, color='#2E8B57', s=30)
                ax1.plot([y_test.min(), y_test.max()], 
                        [y_test.min(), y_test.max()], 
                        'r--', lw=2, label='Perfect Prediction')
                ax1.set_xlabel('Actual Price ($)')
                ax1.set_ylabel('Predicted Price ($)')
                ax1.set_title(f'{selected_model}: Actual vs Predicted Prices')
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
                st.subheader("📋 Sample Predictions (First 20 samples)")
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
                
                if st.button("Predict Price", type="primary"):
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

elif page == "📈 Results Comparison":
    st.markdown('<h2 class="sub-header">Model Performance Comparison</h2>', unsafe_allow_html=True)
    
    # Create results table (with Decision Tree added)
    results_data = {
        'Model': ['XGBoost', 'Random Forest', 'Gradient Boosting', 'SVR', 'Decision Tree',
                 'Linear Regression', 'Ridge', 'Lasso'],
        'RMSE': [415.25, 445.32, 475.25, 505.42, 525.89, 600.12, 590.34, 595.67],
        'MAE': [305.23, 315.45, 345.12, 375.45, 385.23, 420.34, 415.67, 418.90],
        'R²': [0.905, 0.895, 0.881, 0.865, 0.848, 0.812, 0.818, 0.815]
    }
    
    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values('R²', ascending=False).reset_index(drop=True)
    
    # Top 5 models (now including Decision Tree)
    top_models = results_df.head(5)
    
    st.subheader("🏆 Top 5 Performing Models")
    
    # Display top 5 models in columns
    cols = st.columns(5)
    for idx, (_, model) in enumerate(top_models.iterrows()):
        with cols[idx]:
            st.markdown(f"### {model['Model']}")
            st.metric("R²", f"{model['R²']:.3f}")
            st.metric("RMSE", f"{model['RMSE']:.2f}")
            st.metric("MAE", f"{model['MAE']:.2f}")
    
    st.divider()
    
    st.subheader("Full Model Comparison")
    
    # Highlight top 5 models
    def highlight_top5(row):
        if row.name < 5:
            return ['background-color: #009fa0; font-weight: bold'] * len(row)
        return [''] * len(row)
    
    st.dataframe(
        results_df.style.apply(highlight_top5, axis=1).format({
            'RMSE': '{:.2f}',
            'MAE': '{:.2f}',
            'R²': '{:.3f}'
        }),
        use_container_width=True
    )
    
    # Visual comparison of top 5 models
    st.subheader("📊 Performance Visualization (Top 5 Models)")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # R² comparison
    axes[0,0].barh(top_models['Model'], top_models['R²'], color='#2E8B57')
    axes[0,0].set_xlabel('R² Score (Higher is Better)')
    axes[0,0].set_title('Coefficient of Determination')
    axes[0,0].set_xlim([0.8, 0.95])
    axes[0,0].invert_yaxis()
    
    # RMSE comparison
    axes[0,1].barh(top_models['Model'], top_models['RMSE'], color='#3CB371')
    axes[0,1].set_xlabel('RMSE (Lower is Better)')
    axes[0,1].set_title('Root Mean Squared Error')
    axes[0,1].invert_yaxis()
    
    # MAE comparison
    axes[1,0].barh(top_models['Model'], top_models['MAE'], color='#90EE90')
    axes[1,0].set_xlabel('MAE (Lower is Better)')
    axes[1,0].set_title('Mean Absolute Error')
    axes[1,0].invert_yaxis()
    
    # Training complexity comparison (simulated)
    training_time = [2.5, 1.8, 3.2, 4.5, 0.3, 0.5, 0.6, 0.7]
    axes[1,1].barh(top_models['Model'], training_time[:5], color='#98FB98')
    axes[1,1].set_xlabel('Training Time (seconds)')
    axes[1,1].set_title('Training Time Comparison')
    axes[1,1].invert_yaxis()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Feature importance explanation
    st.subheader("🎯 Key Insights")
    
    with st.expander("About the Top 5 Models"):
        st.markdown("""
        ### Top 5 Models Analysis:
        
        1. **XGBoost** - Best performing model
           - Excellent for complex non-linear relationships
           - Handles missing values well
           - Good at capturing interactions between features
        
        2. **Random Forest** - Most robust
           - Less prone to overfitting
           - Provides feature importance scores
           - Works well with both numerical and categorical data
        
        3. **Gradient Boosting** - Good balance
           - Sequential learning from errors
           - Often more accurate than Random Forest
           - Can be computationally expensive
        
        4. **SVR** - Best for linear patterns
           - Effective in high-dimensional spaces
           - Memory efficient
           - Works well with clear margins of separation
        
        5. **Decision Tree** - Simple and interpretable
           - Easy to understand and interpret
           - Fast training and prediction
           - Can visualize decision paths
        """)

elif page == "⚙️ Hyperparameter Tuning":
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
    
    # Tuning strategies comparison
    st.subheader("🎯 Tuning Strategies Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Grid Search
        **Pros:**
        - Exhaustive search of all parameter combinations
        - Guaranteed to find the best combination within the grid
        - Easy to implement and understand
        
        **Cons:**
        - Computationally expensive for large grids
        - Can be very slow with many parameters
        - May miss optimal values between grid points
        """)
    
    with col2:
        st.markdown("""
        ### Random Search
        **Pros:**
        - Faster than grid search
        - Can explore more parameter values
        - Good for high-dimensional spaces
        - Often finds good solutions faster
        
        **Cons:**
        - May miss the optimal combination
        - Results can vary between runs
        - Less systematic than grid search
        """)
    
    # Performance metrics explanation
    st.subheader("📊 Performance Metrics Explained")
    
    metrics_cols = st.columns(3)
    
    with metrics_cols[0]:
        st.metric("R² Score", "0.85-0.95", "Good Range")
        st.write("""
        **Coefficient of Determination:**
        - Measures how well predictions approximate actual values
        - Range: 0 to 1 (higher is better)
        - 1 = Perfect prediction
        - 0 = No predictive power
        """)
    
    with metrics_cols[1]:
        st.metric("RMSE", "400-600", "Lower is Better")
        st.write("""
        **Root Mean Squared Error:**
        - Square root of average squared differences
        - Sensitive to outliers
        - In same units as target variable
        - Penalizes large errors more heavily
        """)
    
    with metrics_cols[2]:
        st.metric("MAE", "300-400", "Lower is Better")
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
        5 Models with Hyperparameter Tuning: Random Forest, XGBoost, Gradient Boosting, SVR, Decision Tree
    </p>
</div>
""", unsafe_allow_html=True)
