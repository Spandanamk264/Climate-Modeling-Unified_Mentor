import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from data_loader import load_data, preprocess_data
from models import ClimateModel
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
    /* Main Background - Very Light Blue */
    .stApp {
        background-color: #E3F2FD; 
        color: #0D47A1;
    }
    
    /* Headers - Deep Royal Blue */
    h1, h2, h3 {
        color: #1565C0; 
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
        font-weight: 600;
    }
    
    /* Metrics Numbers */
    div[data-testid="stMetricValue"] {
        color: #0277BD; /* Ocean Blue */
        font-weight: bold;
    }
    
    /* Metric Labels */
    div[data-testid="stMetricLabel"] {
        color: #455A64;
    }

    /* Sidebar - Slightly Darker Blue-Grey/Ice */
    [data-testid="stSidebar"] {
        background-color: #BBDEFB;
        border-right: 1px solid #90CAF9;
    }
    
    /* Buttons */
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
    
    /* Expander/Cards */
    .streamlit-expanderHeader {
        background-color: #E1F5FE;
        color: #0277BD;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
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
    
    /* Sliders */
    div.stSlider > div[data-baseweb = "slider"] > div > div > div[role="slider"]{
        background-color: #1976D2;
        box-shadow: rgb(14 123 255 / 20%) 0px 0px 0px 0.2rem;
    }
    div.stSlider > div[data-baseweb = "slider"] > div > div > div > div
    {
        background-color: #1976D2;
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
    st.markdown("### ℹ️ About")
    st.info("This advanced dashboard uses Machine Learning to model climate interactions based on CO2, Solar, and Volcanic data.")

# --- Main Dashboard ---
st.title("🌍 Global Climate Change Intelligence (GCCI)")
st.markdown("### 🌊 Advanced Ocean & Atmospheric Analytics Platform")

# KPIs inside a container for styling
with st.container():
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    current_co2 = filtered_df['CO2_Concentration'].iloc[-1]
    prev_co2 = filtered_df['CO2_Concentration'].iloc[-5] # 5 years ago
    current_temp = filtered_df['Global_Temp_Anomaly'].iloc[-1]
    prev_temp = filtered_df['Global_Temp_Anomaly'].iloc[-5]

    with kpi1:
        st.metric("Current CO2 (ppm)", f"{current_co2:.1f}", f"{current_co2 - prev_co2:.1f} (5y)")
    with kpi2:
        st.metric("Global Temp Anomaly (°C)", f"{current_temp:.2f}", f"{current_temp - prev_temp:.2f} (5y)")
    with kpi3:
        st.metric("Solar Irradiance (W/m²)", f"{filtered_df['Solar_Irradiance'].mean():.1f}", "Avg")
    with kpi4:
        st.metric("Data Points Observed", f"{len(filtered_df)}", f"{year_range[1]-year_range[0]} Years")

st.markdown("---")

# --- Tabs for Advanced Organization ---
tab1, tab2, tab3 = st.tabs(["📊 Data Analytics", "🤖 Model Studio", "🔮 Future Simulator"])

# === TAB 1: DATA ANALYTICS ===
with tab1:
    col_main, col_mini = st.columns([3, 1])
    
    with col_main:
        st.markdown("#### Interactive Temporal Analysis")
        # Dual axis plot
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=filtered_df['Year'], y=filtered_df['CO2_Concentration'], name="CO2 (ppm)", line=dict(color='#0277BD', width=3)), secondary_y=False)
        fig.add_trace(go.Scatter(x=filtered_df['Year'], y=filtered_df['Global_Temp_Anomaly'], name="Temp Anomaly (°C)", line=dict(color='#D32F2F', width=3)), secondary_y=True)
        
        fig.update_layout(
            title_text="CO2 Concentration vs Temperature Anomaly", 
            dragmode='zoom', 
            hovermode='x unified', 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(255,255,255,0.5)', # Semi-transparent white
            font=dict(color='#0D47A1'), # Navy Text
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_yaxes(title_text="CO2 (ppm)", secondary_y=False, showgrid=False, title_font=dict(color='#0277BD'))
        fig.update_yaxes(title_text="Temp (°C)", secondary_y=True, showgrid=True, gridcolor='#B3E5FC', title_font=dict(color='#D32F2F'))
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_mini:
        st.markdown("#### Correlations")
        st.caption("Correlation Matrix between key climate variables.")
        corr = filtered_df[['CO2_Concentration', 'Global_Temp_Anomaly', 'Solar_Irradiance', 'Volcanic_Activity']].corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='Blues', aspect='auto')
        fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#0D47A1'))
        st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("#### Distribution Analysis")
    fig_hist = px.histogram(filtered_df, x="Global_Temp_Anomaly", nbins=30, marginal="box", title="Temperature Anomaly Distribution", color_discrete_sequence=['#0288D1'])
    fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#0D47A1'))
    fig_hist.update_xaxes(showgrid=True, gridcolor='#B3E5FC')
    fig_hist.update_yaxes(showgrid=True, gridcolor='#B3E5FC')
    st.plotly_chart(fig_hist, use_container_width=True)

# === TAB 2: MODEL STUDIO ===
with tab2:
    st.subheader("Machine Learning Model Training")
    st.caption("Train predictive models on the historical dataset to understand drivers of climate change.")
    
    model_col1, model_col2 = st.columns([1, 3])
    
    with model_col1:
        st.markdown("#### Configuration")
        with st.expander("Model Settings", expanded=True):
            model_type = st.radio("Select Algorithm", ["Random Forest Regressor", "Linear Regression"])
            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2)
            
            train_btn = st.button("🚀 Train Model")
    
    with model_col2:
        if train_btn:
            with st.spinner("Training advanced model..."):
                X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df, test_size=test_size)
                
                model_key = 'random_forest' if "Random" in model_type else 'linear'
                model = ClimateModel(model_key)
                model.train(X_train, y_train)
                metrics = model.evaluate(X_test, y_test)
                
                # Metrics Row
                m1, m2, m3 = st.columns(3)
                m1.metric("R² Score (Accuracy)", f"{metrics['R2']:.4f}", delta_color="normal")
                m2.metric("MAE", f"{metrics['MAE']:.4f}", delta_color="inverse")
                m3.metric("MSE", f"{metrics['MSE']:.4f}", delta_color="inverse")
                
                # Visuals
                fig_pred = px.scatter(x=y_test, y=metrics['predictions'], labels={'x': 'Actual Temperature', 'y': 'Predicted Temperature'}, title="Actual vs Predicted Precision")
                fig_pred.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Ideal Fit', line=dict(dash='dash', color='black')))
                fig_pred.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#0D47A1'))
                fig_pred.update_xaxes(showgrid=True, gridcolor='#B3E5FC')
                fig_pred.update_yaxes(showgrid=True, gridcolor='#B3E5FC')
                st.plotly_chart(fig_pred, use_container_width=True)
                
                if model_key == 'random_forest':
                    importances = model.model.feature_importances_
                    fig_feat = px.bar(x=feature_names, y=importances, title="Feature Importance (Drivers)", labels={'x':'Feature', 'y':'Importance'}, color=importances, color_continuous_scale='Blues')
                    fig_feat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#0D47A1'))
                    st.plotly_chart(fig_feat, use_container_width=True)
                
                # Save model for simulation tab
                st.session_state['trained_model'] = model
                st.session_state['scaler'] = scaler
                st.success("Model trained successfully and ready for simulation!")

# === TAB 3: FUTURE SIMULATOR ===
with tab3:
    st.subheader("🌍 2050 Climate Simulator")
    st.markdown("Use the controls below to simulate future environmental conditions and predict global temperature anomalies.")
    
    if 'trained_model' not in st.session_state:
        st.info("👈 Please train a model in the 'Model Studio' tab first to enable the simulator.")
    else:
        # Layout: Inputs on left, Gauge on right
        sim_col1, sim_col2 = st.columns([1, 1])
        
        with sim_col1:
            st.markdown("#### Environmental Drivers")
            st.markdown("Adjust these sliders to simulate a future scenario:")
            sim_co2 = st.slider("CO2 Concentration (ppm)", 300.0, 1000.0, 420.0, help="Pre-industrial: ~280ppm. Current: ~420ppm.")
            sim_solar = st.slider("Solar Irradiance (W/m²)", 1355.0, 1370.0, 1361.0, help="Energy form the sun.")
            sim_volcanic = st.slider("Volcanic Activity Index", -1.0, 0.5, 0.0, help="Negative values indicate cooling ash.")
            sim_sealevel = st.slider("Projected Sea Level Rise (mm)", 0.0, 500.0, 100.0) # Dummy for model input structure
            
        with sim_col2:
            model = st.session_state['trained_model']
            scaler = st.session_state['scaler']
            
            # Predict
            input_vector = np.array([[sim_co2, sim_solar, sim_volcanic, sim_sealevel]])
            input_scaled = scaler.transform(input_vector)
            prediction = model.predict(input_scaled)[0]
            
            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = prediction,
                delta = {'reference': 1.5, 'position': "top"},
                title = {'text': "Predicted Temp Anomaly (°C)", 'font': {'size': 20, 'color': '#0D47A1'}},
                gauge = {
                    'axis': {'range': [-1, 5], 'tickwidth': 1, 'tickcolor': "#0D47A1"},
                    'bar': {'color': "#1976D2"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#0D47A1",
                    'steps': [
                        {'range': [-1, 1.5], 'color': "#E3F2FD"}, # Very Light Blue
                        {'range': [1.5, 5], 'color': "#FFEBEE"}], # Very Light Red
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 1.5}}))
            
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#0D47A1'))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.markdown("### Diagnosis")
            if prediction > 1.5:
                st.error(f"💥 **CRITICAL WARNING**: This scenario predicts a warming of **{prediction:.2f}°C**, exceeding the Paris Agreement 1.5°C limit.")
            else:
                st.success(f"✅ **SAFE**: This scenario predicts a warming of **{prediction:.2f}°C**, which is within sustainable limits.")

st.markdown("---")
st.caption("ClimateAI v2.2 | Ocean Blue Edition | Advanced Analytics Platform")
