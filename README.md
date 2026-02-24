# 🌍 Climate Change Modeling & Predictive Intelligence

> **What happens when you combine Machine Learning with Climate Science?**
>
> This project is a comprehensive **Climate Change Modeling System** developed during my **Data Science Internship at Unified Mentor**. It represents one of my project, where I explore the intersection of environmental physics and predictive analytics.

---

## 📖 The Story Behind the Data
I built this system to go beyond simply analyzing historical climate data. It processes **120+ years** of environmental records across **7 climate variables**, engineers **10 intelligent features** (including rolling averages and lagged indicators), and trains **5 different ML models** — Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost.

The most striking moment came when the model independently reinforced conclusions long established by climate researchers. Among all environmental drivers analyzed, **CO₂ concentration** emerged as the single most dominant predictor of temperature anomalies. The projections were consistent. Data, when modeled responsibly, tells a powerful story.

---

## 🚀 Key Features
- **Multi-Model Pipeline**: Implementation and comparison of 5 algorithms (Linear, Tree-based, Gradient Boosting, and XGBoost).
- **Advanced Feature Engineering**: 3yr/5yr rolling averages, 1yr/2yr lags, and rate-of-change indicators.
- **Robust Evaluation**: k-fold cross-validation, residual analysis (scatter + histogram), and R² monitoring (hitting up to **0.87 accuracy**).
- **Interactive Web Dashboard**: A Streamlit-based interface for real-time scenario simulation and data visualization.
- **Scenario Simulator**: A "2050 Simulator" that flags when projections exceed the **Paris Agreement's 1.5°C threshold**.

---

## 🛠️ Tech Stack
- **Languages**: Python
- **Machine Learning**: Scikit-Learn, XGBoost
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Deployment**: Streamlit

---

## 📂 Project Structure
- `data/`: Contains the generated dataset (`climate_data.csv`).
- `src/`: Core source code.
  - `app.py`: The Main Streamlit Dashboard.
  - `models.py`: Model wrapper and comparison engine.
  - `data_loader.py`: Preprocessing and advanced feature engineering.
  - `data_generator.py`: Realistic synthetic data simulator (Physical principle based).
  - `main.py`: CLI orchestration script.
- `results/`: Output directory for plots and metrics.
- `notebooks/`: Interactive analysis workspace.

---

## ⚙️ Setup & Execution

1.  **Run the Setup Script** (One-click installation):
    ```powershell
    .\setup_project.bat
    ```
    *This will create the environment, install dependencies, and run the pipeline.*

2.  **Run the Dashboard**:
    ```bash
    streamlit run src/app.py
    ```

---

## 🙏 Acknowledgements
A huge thank you to **Unified Mentor** for providing this opportunity. Each project in this internship has challenged me to think deeper, build smarter, and approach real-world problems with greater responsibility.

Here’s to building technology that informs, empowers, and serves a greater purpose. 🚀

---
*ClimateAI v3.0 | Developed by Spandana Mahadevappa Kandagal*
