import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Oil Palm Price Predictor",
    page_icon="🌴",
    layout="wide"
)

# Title and description
st.title("🌴 Oil Palm Price Prediction System")
st.markdown("""
This application predicts oil palm prices based on weather conditions, production indices, 
export data, and currency exchange rates using machine learning models.
""")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Train Models", "Make Predictions", "Model Comparison"])

# Helper function to load data
@st.cache_data
def load_data(uploaded_files):
    """Load and merge all uploaded data files"""
    if len(uploaded_files) < 7:
        st.error("Please upload all 7 required files!")
        return None
    
    try:
        # Load each file
        data_dict = {}
        for file in uploaded_files:
            df = pd.read_csv(file)
            data_dict[file.name] = df
        
        return data_dict
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def preprocess_data(data_dict):
    """Preprocess and merge all datasets"""
    try:
        # Load weather data
        weather_df = data_dict.get('weather.csv')
        if weather_df is None:
            st.error("weather.csv not found!")
            return None, None, None, None
        
        weather_df['Date'] = pd.to_datetime(weather_df['Date'])
        
        # Load price data
        price_dfs = []
        for year in ['2020', '2021', '2022']:
            key = f'price{year}.csv'
            if key in data_dict:
                price_df = data_dict[key]
                price_df['Date'] = pd.to_datetime(price_df['Date'], errors='coerce')
                price_df.dropna(subset=['Date'], inplace=True)
                
                # Expand to daily frequency
                price_df = price_df.set_index('Date')
                price_expanded = price_df.resample('D').ffill().reset_index()
                price_dfs.append(price_expanded)
        
        # Combine all price data
        if price_dfs:
            price_combined = pd.concat(price_dfs, ignore_index=True)
        else:
            st.error("No price data found!")
            return None, None, None, None
        
        # Load and expand IPI data
        ipi_df = data_dict.get('ipi.csv')
        if ipi_df is not None:
            ipi_df['Date'] = pd.to_datetime(ipi_df['Date'], errors='coerce')
            ipi_df.dropna(subset=['Date'], inplace=True)
            ipi_df = ipi_df.set_index('Date')
            ipi_expanded = ipi_df.resample('D').ffill().reset_index()
        else:
            st.warning("IPI data not found, continuing without it")
            ipi_expanded = None
        
        # Load and expand exchange data
        exchange_df = data_dict.get('exchange.csv')
        if exchange_df is not None:
            exchange_df['Date'] = pd.to_datetime(exchange_df['Date'], errors='coerce')
            exchange_df.dropna(subset=['Date'], inplace=True)
            exchange_df = exchange_df.set_index('Date')
            exchange_expanded = exchange_df.resample('D').ffill().reset_index()
        else:
            st.warning("Exchange data not found, continuing without it")
            exchange_expanded = None
        
        # Load and expand export data
        export_df = data_dict.get('export.csv')
        if export_df is not None:
            export_df['Date'] = pd.to_datetime(export_df['Date'], errors='coerce')
            export_df.dropna(subset=['Date'], inplace=True)
            export_df = export_df.set_index('Date')
            export_expanded = export_df.resample('D').ffill().reset_index()
        else:
            st.warning("Export data not found, continuing without it")
            export_expanded = None
        
        # Merge all data
        df = weather_df.copy()
        
        if ipi_expanded is not None:
            df = pd.merge(df, ipi_expanded, on='Date', how='left')
        
        df = pd.merge(df, price_combined, on='Date', how='left')
        
        if exchange_expanded is not None:
            df = pd.merge(df, exchange_expanded, on='Date', how='left')
        
        if export_expanded is not None:
            df = pd.merge(df, export_expanded, on='Date', how='left')
        
        # Convert Export Number to numeric if it exists
        if 'Export Number (in Tonnes)' in df.columns:
            df['Export Number (in Tonnes)'] = pd.to_numeric(
                df['Export Number (in Tonnes)'], errors='coerce'
            )
        
        # Handle missing values with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # Feature engineering
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        
        # Prepare features and target
        df_processed = df.drop(columns=['Date'])
        
        if 'Price' not in df_processed.columns:
            st.error("Price column not found in data!")
            return None, None, None, None
        
        X = df_processed.drop(columns=['Price'])
        y = df_processed['Price']
        
        return X, y, df, X.columns.tolist()
    
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None

# PAGE 1: Train Models
if page == "Train Models":
    st.header("📊 Train Machine Learning Models")
    
    st.markdown("""
    ### Required Data Files (CSV format):
    1. **weather.csv** - Weather data with Date column
    2. **price2020.csv** - Price data for 2020
    3. **price2021.csv** - Price data for 2021
    4. **price2022.csv** - Price data for 2022
    5. **ipi.csv** - Industrial Production Index
    6. **exchange.csv** - Currency exchange rates
    7. **export.csv** - Export data
    """)
    
    uploaded_files = st.file_uploader(
        "Upload all 7 CSV files",
        type=['csv'],
        accept_multiple_files=True
    )
    
    if uploaded_files and len(uploaded_files) == 7:
        with st.spinner("Loading and preprocessing data..."):
            data_dict = {}
            for file in uploaded_files:
                df = pd.read_csv(file)
                data_dict[file.name] = df
            
            X, y, df_full, feature_names = preprocess_data(data_dict)
        
        if X is not None and y is not None:
            st.success(f"✅ Data loaded successfully! Total samples: {len(X)}")
            
            # Display data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Features", X.shape[1])
            with col2:
                st.metric("Total Samples", X.shape[0])
            with col3:
                st.metric("Price Range", f"RM{y.min():.2f} - RM{y.max():.2f}")
            
            # Show sample data
            with st.expander("📋 View Sample Data"):
                st.dataframe(df_full.head(10))
            
            # Train models
            if st.button("🚀 Train Models", type="primary"):
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
                from sklearn.feature_selection import RFE
                from sklearn.linear_model import LinearRegression, Ridge, Lasso
                from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                from sklearn.tree import DecisionTreeRegressor
                from sklearn.svm import SVR
                from sklearn.neural_network import MLPRegressor
                from sklearn.pipeline import make_pipeline
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                from xgboost import XGBRegressor
                
                with st.spinner("Training models... This may take a few minutes."):
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # Feature selection with RFE
                    scaler = StandardScaler()
                    estimator = LinearRegression()
                    
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    rfe = RFE(estimator=estimator, n_features_to_select=10)
                    rfe.fit(X_train_scaled, y_train)
                    
                    selected_features = X.columns[rfe.support_].tolist()
                    st.info(f"🎯 Selected Features: {', '.join(selected_features)}")
                    
                    X_train_rfe = rfe.transform(X_train_scaled)
                    X_test_rfe = rfe.transform(X_test_scaled)
                    
                    # Train multiple models
                    results = []
                    trained_models = {}
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    models = {
                        'Linear Regression': LinearRegression(),
                        'Ridge': Ridge(alpha=1.0),
                        'Lasso': Lasso(alpha=0.01),
                        'XGBoost': XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, 
                                               subsample=0.8, colsample_bytree=0.8, random_state=42),
                        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=5, 
                                                               min_samples_split=5, random_state=42),
                        'Gradient Boosting': GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, 
                                                                       max_depth=3, random_state=42),
                        'Decision Tree': DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state=42),
                        'SVR': make_pipeline(SVR(kernel='rbf', C=100, epsilon=0.1)),
                        'MLP': make_pipeline(MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, 
                                                         random_state=42, early_stopping=True))
                    }
                    
                    total_models = len(models)
                    for idx, (name, model) in enumerate(models.items()):
                        status_text.text(f"Training {name}...")
                        
                        # Train
                        if name in ['SVR', 'MLP']:
                            model.fit(X_train_rfe, y_train)
                            pred = model.predict(X_test_rfe)
                        else:
                            model.fit(X_train_rfe, y_train)
                            pred = model.predict(X_test_rfe)
                        
                        # Evaluate
                        rmse = np.sqrt(mean_squared_error(y_test, pred))
                        mae = mean_absolute_error(y_test, pred)
                        r2 = r2_score(y_test, pred)
                        
                        results.append({
                            'Model': name,
                            'RMSE': rmse,
                            'MAE': mae,
                            'R²': r2
                        })
                        
                        trained_models[name] = model
                        
                        progress_bar.progress((idx + 1) / total_models)
                    
                    status_text.text("Training complete!")
                    
                    # Display results
                    results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
                    
                    st.success("✅ All models trained successfully!")
                    st.subheader("📈 Model Performance Comparison")
                    
                    # Highlight best model
                    st.dataframe(
                        results_df.style.highlight_max(subset=['R²'], color='lightgreen')
                                       .highlight_min(subset=['RMSE', 'MAE'], color='lightgreen')
                                       .format({'RMSE': '{:.4f}', 'MAE': '{:.4f}', 'R²': '{:.4f}'})
                    )
                    
                    # Save best model
                    best_model_name = results_df.iloc[0]['Model']
                    best_model = trained_models[best_model_name]
                    
                    # Save artifacts
                    model_artifacts = {
                        'model': best_model,
                        'scaler': scaler,
                        'rfe': rfe,
                        'feature_names': feature_names,
                        'selected_features': selected_features,
                        'model_name': best_model_name
                    }
                    
                    with open('/tmp/best_model.pkl', 'wb') as f:
                        pickle.dump(model_artifacts, f)
                    
                    st.session_state['model_trained'] = True
                    st.session_state['best_model_name'] = best_model_name
                    st.session_state['results_df'] = results_df
                    
                    st.success(f"🏆 Best Model: **{best_model_name}** (R² = {results_df.iloc[0]['R²']:.4f})")
                    st.info("Model saved! Go to 'Make Predictions' to use it.")

# PAGE 2: Make Predictions
elif page == "Make Predictions":
    st.header("🔮 Make Price Predictions")
    
    if not os.path.exists('/tmp/best_model.pkl'):
        st.warning("⚠️ No trained model found. Please train a model first in the 'Train Models' page.")
    else:
        # Load model
        with open('/tmp/best_model.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        
        model = artifacts['model']
        scaler = artifacts['scaler']
        rfe = artifacts['rfe']
        feature_names = artifacts['feature_names']
        selected_features = artifacts['selected_features']
        model_name = artifacts['model_name']
        
        st.success(f"✅ Loaded trained model: **{model_name}**")
        st.info(f"🎯 Selected Features: {', '.join(selected_features)}")
        
        st.markdown("### Enter Feature Values")
        
        # Create input form
        input_data = {}
        
        # Organize inputs into columns
        col1, col2, col3 = st.columns(3)
        
        num_features = len(feature_names)
        features_per_col = (num_features + 2) // 3
        
        with col1:
            for i, feature in enumerate(feature_names[:features_per_col]):
                if 'Year' in feature:
                    input_data[feature] = st.number_input(f"{feature}", value=2023, step=1)
                elif 'Month' in feature:
                    input_data[feature] = st.number_input(f"{feature}", value=6, min_value=1, max_value=12, step=1)
                elif 'Day' in feature:
                    input_data[feature] = st.number_input(f"{feature}", value=15, min_value=1, max_value=31, step=1)
                else:
                    input_data[feature] = st.number_input(f"{feature}", value=0.0)
        
        with col2:
            for feature in feature_names[features_per_col:2*features_per_col]:
                if 'Year' in feature:
                    input_data[feature] = st.number_input(f"{feature}", value=2023, step=1)
                elif 'Month' in feature:
                    input_data[feature] = st.number_input(f"{feature}", value=6, min_value=1, max_value=12, step=1)
                elif 'Day' in feature:
                    input_data[feature] = st.number_input(f"{feature}", value=15, min_value=1, max_value=31, step=1)
                else:
                    input_data[feature] = st.number_input(f"{feature}", value=0.0)
        
        with col3:
            for feature in feature_names[2*features_per_col:]:
                if 'Year' in feature:
                    input_data[feature] = st.number_input(f"{feature}", value=2023, step=1)
                elif 'Month' in feature:
                    input_data[feature] = st.number_input(f"{feature}", value=6, min_value=1, max_value=12, step=1)
                elif 'Day' in feature:
                    input_data[feature] = st.number_input(f"{feature}", value=15, min_value=1, max_value=31, step=1)
                else:
                    input_data[feature] = st.number_input(f"{feature}", value=0.0)
        
        if st.button("🎯 Predict Price", type="primary"):
            # Prepare input
            input_df = pd.DataFrame([input_data])
            
            # Scale and select features
            input_scaled = scaler.transform(input_df)
            input_rfe = rfe.transform(input_scaled)
            
            # Make prediction
            prediction = model.predict(input_rfe)[0]
            
            # Display result
            st.markdown("---")
            st.subheader("Prediction Result")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.metric(
                    label="Predicted Oil Palm Price",
                    value=f"RM {prediction:.2f}",
                    delta=None
                )
            
            st.balloons()

# PAGE 3: Model Comparison
elif page == "Model Comparison":
    st.header("📊 Model Performance Comparison")
    
    if 'results_df' not in st.session_state:
        st.warning("⚠️ No model results found. Please train models first in the 'Train Models' page.")
    else:
        results_df = st.session_state['results_df']
        
        st.subheader("Performance Metrics Table")
        st.dataframe(
            results_df.style.highlight_max(subset=['R²'], color='lightgreen')
                           .highlight_min(subset=['RMSE', 'MAE'], color='lightgreen')
                           .format({'RMSE': '{:.4f}', 'MAE': '{:.4f}', 'R²': '{:.4f}'}),
            use_container_width=True
        )
        
        # Visualizations
        import plotly.graph_objects as go
        import plotly.express as px
        
        st.subheader("📈 Visual Comparison")
        
        # R² Score Comparison
        fig_r2 = px.bar(
            results_df,
            x='Model',
            y='R²',
            title='R² Score by Model',
            color='R²',
            color_continuous_scale='Greens',
            text='R²'
        )
        fig_r2.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig_r2.update_layout(height=500)
        st.plotly_chart(fig_r2, use_container_width=True)
        
        # RMSE and MAE Comparison
        col1, col2 = st.columns(2)
        
        with col1:
            fig_rmse = px.bar(
                results_df,
                x='Model',
                y='RMSE',
                title='RMSE by Model',
                color='RMSE',
                color_continuous_scale='Reds_r',
                text='RMSE'
            )
            fig_rmse.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig_rmse.update_layout(height=400)
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        with col2:
            fig_mae = px.bar(
                results_df,
                x='Model',
                y='MAE',
                title='MAE by Model',
                color='MAE',
                color_continuous_scale='Reds_r',
                text='MAE'
            )
            fig_mae.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig_mae.update_layout(height=400)
            st.plotly_chart(fig_mae, use_container_width=True)
        
        # Best model info
        best_model = results_df.iloc[0]
        st.success(f"""
        ### 🏆 Best Performing Model
        - **Model**: {best_model['Model']}
        - **R² Score**: {best_model['R²']:.4f}
        - **RMSE**: {best_model['RMSE']:.4f}
        - **MAE**: {best_model['MAE']:.4f}
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Oil Palm Price Prediction System | Developed with Streamlit</p>
</div>
""", unsafe_allow_html=True)
