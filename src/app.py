import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from data_loader import load_data, preprocess_data, feature_engineering
from models import ClimateModel, compare_models
from data_generator import generate_synthetic_data

# --- Page Configuration ---
st.set_page_config(
    page_title="ClimateAI - Advanced Modeling",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for "Light Blue Professional" Look ---
st.markdown("""
<style>
    .stApp {
        background-color: #E3F2FD; 
        color: #0D47A1;
    }
    h1, h2, h3 {
        color: #1565C0; 
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #0277BD;
        font-weight: bold;
    }
    div[data-testid="stMetricLabel"] {
        color: #455A64;
    }
    [data-testid="stSidebar"] {
        background-color: #BBDEFB;
        border-right: 1px solid #90CAF9;
    }
    .stButton>button {
        color: white;
        background-color: #1976D2;
        border-radius: 8px;
        border: none;
        box-shadow: 0px 2px 4px rgba(0,0,0,0.1);
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0D47A1;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF;
        border-radius: 4px;
        color: #1565C0;
        border: 1px solid #BBDEFB;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2196F3;
        color: white !important;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Handling ---
DATA_PATH = os.path.join("data", "climate_data.csv")

@st.cache_data
def get_data():
    if not os.path.exists(DATA_PATH):
        return generate_synthetic_data(DATA_PATH)
    return load_data(DATA_PATH)

df = get_data()

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/869/869869.png", width=80)
    st.title("ClimateAI Control")
    st.markdown("---")

    st.subheader("Data Filters")
    year_range = st.slider("Select Year Range",
                           int(df['Year'].min()),
                           int(df['Year'].max()),
                           (1950, 2023))

    filtered_df = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]

    st.markdown("---")
    st.markdown("### About")
    st.info("Advanced ML dashboard for climate modeling. Supports 5 algorithms, cross-validation, feature engineering, and scenario analysis.")

# --- Main Dashboard ---
st.title("🌍 Global Climate Change Intelligence (GCCI)")
st.markdown("### Advanced Ocean & Atmospheric Analytics Platform")

# KPIs
with st.container():
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    current_co2 = filtered_df['CO2_Concentration'].iloc[-1]
    prev_co2 = filtered_df['CO2_Concentration'].iloc[-5]
    current_temp = filtered_df['Global_Temp_Anomaly'].iloc[-1]
    prev_temp = filtered_df['Global_Temp_Anomaly'].iloc[-5]

    with kpi1:
        st.metric("Current CO2 (ppm)", f"{current_co2:.1f}", f"{current_co2 - prev_co2:.1f} (5y)")
    with kpi2:
        st.metric("Temp Anomaly (C)", f"{current_temp:.2f}", f"{current_temp - prev_temp:.2f} (5y)")
    with kpi3:
        st.metric("Solar Irradiance (W/m2)", f"{filtered_df['Solar_Irradiance'].mean():.1f}", "Avg")
    with kpi4:
        st.metric("Data Points", f"{len(filtered_df)}", f"{year_range[1]-year_range[0]} Years")

st.markdown("---")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Analytics", "🤖 Model Studio", "📈 Model Comparison", "🔮 Future Simulator"])

# === TAB 1: DATA ANALYTICS ===
with tab1:
    col_main, col_mini = st.columns([3, 1])

    with col_main:
        st.markdown("#### Interactive Temporal Analysis")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=filtered_df['Year'], y=filtered_df['CO2_Concentration'], name="CO2 (ppm)", line=dict(color='#0277BD', width=3)), secondary_y=False)
        fig.add_trace(go.Scatter(x=filtered_df['Year'], y=filtered_df['Global_Temp_Anomaly'], name="Temp Anomaly (C)", line=dict(color='#D32F2F', width=3)), secondary_y=True)
        fig.update_layout(title_text="CO2 vs Temperature Anomaly", dragmode='zoom', hovermode='x unified',
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.5)',
                          font=dict(color='#0D47A1'), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_yaxes(title_text="CO2 (ppm)", secondary_y=False, showgrid=False, title_font=dict(color='#0277BD'))
        fig.update_yaxes(title_text="Temp (C)", secondary_y=True, showgrid=True, gridcolor='#B3E5FC', title_font=dict(color='#D32F2F'))
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_mini:
        st.markdown("#### Correlations")
        corr_cols = [c for c in ['CO2_Concentration', 'Global_Temp_Anomaly', 'Solar_Irradiance', 'Volcanic_Activity', 'Precipitation'] if c in filtered_df.columns]
        corr = filtered_df[corr_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='Blues', aspect='auto')
        fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#0D47A1'))
        st.plotly_chart(fig_corr, use_container_width=True)

    # Feature Engineering Preview
    st.markdown("#### Feature Engineering Preview")
    st.caption("Showing engineered features: rolling averages, lagged variables, rate of change.")
    fe_df = feature_engineering(filtered_df)
    st.dataframe(fe_df[['Year', 'CO2_Concentration', 'CO2_Rolling_3yr', 'CO2_Rolling_5yr', 'CO2_Lag_1yr', 'CO2_Rate_of_Change']].tail(15), use_container_width=True)

    st.markdown("#### Distribution Analysis")
    fig_hist = px.histogram(filtered_df, x="Global_Temp_Anomaly", nbins=30, marginal="box", title="Temperature Anomaly Distribution", color_discrete_sequence=['#0288D1'])
    fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#0D47A1'))
    fig_hist.update_xaxes(showgrid=True, gridcolor='#B3E5FC')
    fig_hist.update_yaxes(showgrid=True, gridcolor='#B3E5FC')
    st.plotly_chart(fig_hist, use_container_width=True)

# === TAB 2: MODEL STUDIO ===
with tab2:
    st.subheader("Machine Learning Model Training")
    st.caption("Train a single model with cross-validation and view residual analysis.")

    model_col1, model_col2 = st.columns([1, 3])

    with model_col1:
        st.markdown("#### Configuration")
        with st.expander("Model Settings", expanded=True):
            model_type = st.radio("Select Algorithm", [
                "Random Forest Regressor",
                "Linear Regression",
                "Decision Tree",
                "Gradient Boosting",
                "XGBoost"
            ])
            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2)
            cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
            train_btn = st.button("🚀 Train Model")

    with model_col2:
        if train_btn:
            with st.spinner("Training model with cross-validation..."):
                X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df, test_size=test_size)

                model_map = {
                    "Random Forest Regressor": "random_forest",
                    "Linear Regression": "linear",
                    "Decision Tree": "decision_tree",
                    "Gradient Boosting": "gradient_boosting",
                    "XGBoost": "xgboost"
                }
                model_key = model_map[model_type]
                model = ClimateModel(model_key)
                model.train(X_train, y_train)
                metrics = model.evaluate(X_test, y_test)

                # Cross-Validation
                cv_results = model.cross_validate(X_train, y_train, cv=cv_folds)

                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("R2 Score", f"{metrics['R2']:.4f}")
                m2.metric("MAE", f"{metrics['MAE']:.4f}")
                m3.metric("MSE", f"{metrics['MSE']:.4f}")
                m4.metric(f"CV R2 ({cv_folds}-fold)", f"{cv_results['cv_mean']:.4f} +/- {cv_results['cv_std']:.4f}")

                # Actual vs Predicted Plot
                st.markdown("##### Actual vs Predicted")
                fig_pred = px.scatter(x=y_test, y=metrics['predictions'], labels={'x': 'Actual', 'y': 'Predicted'}, title="Actual vs Predicted")
                fig_pred.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Ideal', line=dict(dash='dash', color='black')))
                fig_pred.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#0D47A1'))
                fig_pred.update_xaxes(showgrid=True, gridcolor='#B3E5FC')
                fig_pred.update_yaxes(showgrid=True, gridcolor='#B3E5FC')
                st.plotly_chart(fig_pred, use_container_width=True)

                # Residual Plot
                st.markdown("##### Residual Analysis")
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    fig_res = px.scatter(x=metrics['predictions'], y=metrics['residuals'], labels={'x': 'Predicted', 'y': 'Residual'}, title="Residual Plot")
                    fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                    fig_res.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#0D47A1'))
                    st.plotly_chart(fig_res, use_container_width=True)
                with res_col2:
                    fig_res_hist = px.histogram(x=metrics['residuals'], nbins=20, title="Residual Distribution", color_discrete_sequence=['#0288D1'])
                    fig_res_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#0D47A1'))
                    st.plotly_chart(fig_res_hist, use_container_width=True)

                # Feature Importance (tree-based models only)
                if hasattr(model.model, 'feature_importances_'):
                    st.markdown("##### Feature Importance")
                    importances = model.model.feature_importances_
                    fig_feat = px.bar(x=feature_names, y=importances, title="Feature Importance", labels={'x': 'Feature', 'y': 'Importance'}, color=importances, color_continuous_scale='Blues')
                    fig_feat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#0D47A1'))
                    st.plotly_chart(fig_feat, use_container_width=True)

                # Save to session
                st.session_state['trained_model'] = model
                st.session_state['scaler'] = scaler
                st.session_state['feature_names'] = feature_names
                st.success("Model trained and saved!")

# === TAB 3: MODEL COMPARISON ===
with tab3:
    st.subheader("Multi-Model Comparison")
    st.caption("Train ALL 5 models simultaneously and compare their performance to find the best one.")

    if st.button("🏁 Compare All Models"):
        with st.spinner("Training 5 models... This may take a moment."):
            X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
            results = compare_models(X_train, y_train, X_test, y_test)

            # Comparison Table
            comparison_data = []
            for name, res in results.items():
                comparison_data.append({
                    'Model': name,
                    'R2 Score': round(res['R2'], 4),
                    'MAE': round(res['MAE'], 4),
                    'MSE': round(res['MSE'], 4),
                    'CV Mean R2': round(res['CV_Mean_R2'], 4),
                    'CV Std R2': round(res['CV_Std_R2'], 4)
                })

            comp_df = pd.DataFrame(comparison_data).sort_values('R2 Score', ascending=False)
            best_model_name = comp_df.iloc[0]['Model']

            st.markdown(f"### 🏆 Best Model: **{best_model_name}**")
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Bar chart comparison
            fig_comp = px.bar(comp_df, x='Model', y='R2 Score', color='R2 Score', title="R2 Score Comparison", color_continuous_scale='Blues')
            fig_comp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#0D47A1'))
            st.plotly_chart(fig_comp, use_container_width=True)

            # Cross-validation comparison
            fig_cv = go.Figure()
            for name, res in results.items():
                fig_cv.add_trace(go.Bar(name=name, x=[name], y=[res['CV_Mean_R2']], error_y=dict(type='data', array=[res['CV_Std_R2']])))
            fig_cv.update_layout(title="Cross-Validation R2 Scores (with Std Dev)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#0D47A1'), showlegend=False)
            st.plotly_chart(fig_cv, use_container_width=True)

            # Save best model
            best_key = [k for k, v in ClimateModel.SUPPORTED_MODELS.items() if v == best_model_name][0]
            st.session_state['trained_model'] = results[best_model_name]['model_obj']
            st.session_state['scaler'] = scaler
            st.session_state['feature_names'] = feature_names
            st.success(f"Best model ({best_model_name}) saved for simulation!")

# === TAB 4: FUTURE SIMULATOR ===
with tab4:
    st.subheader("🌍 2050 Climate Simulator")
    st.markdown("Adjust the drivers below to simulate future scenarios and predict global temperature anomalies.")

    if 'trained_model' not in st.session_state:
        st.info("Please train a model in the 'Model Studio' or 'Model Comparison' tab first.")
    else:
        sim_col1, sim_col2 = st.columns([1, 1])

        with sim_col1:
            st.markdown("#### Environmental Drivers")
            sim_co2 = st.slider("CO2 Concentration (ppm)", 300.0, 1000.0, 420.0, help="Pre-industrial: ~280ppm. Current: ~420ppm.")
            sim_solar = st.slider("Solar Irradiance (W/m2)", 1355.0, 1370.0, 1361.0)
            sim_volcanic = st.slider("Volcanic Activity Index", -1.0, 0.5, 0.0)
            sim_precipitation = st.slider("Precipitation (mm)", 800.0, 1500.0, 1060.0)
            sim_sealevel = st.slider("Sea Level Rise (mm)", 0.0, 500.0, 100.0)

        with sim_col2:
            model = st.session_state['trained_model']
            scaler = st.session_state['scaler']
            feature_names = st.session_state.get('feature_names', [])

            # Build input matching the feature count
            n_features = len(feature_names)
            # Base features: CO2, Solar, Volcanic, Precipitation, SeaLevel + engineered features
            base_vals = [sim_co2, sim_solar, sim_volcanic, sim_precipitation, sim_sealevel]
            # Engineered features: CO2_Rolling_3yr, CO2_Rolling_5yr, CO2_Lag_1yr, CO2_Lag_2yr, CO2_Rate_of_Change
            engineered_vals = [sim_co2, sim_co2, sim_co2 * 0.99, sim_co2 * 0.98, sim_co2 * 0.01]
            all_vals = base_vals + engineered_vals
            # Take only as many as needed
            input_data = np.array([all_vals[:n_features]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]

            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                delta={'reference': 1.5, 'position': "top"},
                title={'text': "Predicted Temp Anomaly (C)", 'font': {'size': 20, 'color': '#0D47A1'}},
                gauge={
                    'axis': {'range': [-1, 5], 'tickwidth': 1, 'tickcolor': "#0D47A1"},
                    'bar': {'color': "#1976D2"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#0D47A1",
                    'steps': [
                        {'range': [-1, 1.5], 'color': "#E3F2FD"},
                        {'range': [1.5, 5], 'color': "#FFEBEE"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 1.5}}))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#0D47A1'))
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown("### Diagnosis")
            if prediction > 1.5:
                st.error(f"CRITICAL: Predicted warming of **{prediction:.2f}C** exceeds the Paris Agreement 1.5C limit.")
            elif prediction > 1.0:
                st.warning(f"CAUTION: Predicted warming of **{prediction:.2f}C** is approaching the 1.5C limit.")
            else:
                st.success(f"SAFE: Predicted warming of **{prediction:.2f}C** is within sustainable limits.")

st.markdown("---")
st.caption("ClimateAI v3.0 | Full Compliance Edition | 5 Models + Cross-Validation + Feature Engineering")
