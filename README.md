# Climate Change Modeling Project

## Project Overview
This project develops a machine learning model to predict Global Temperature Anomalies based on environmental factors such as CO2 concentration, solar irradiance, and volcanic activity. It demonstrates the full data science lifecycle: data generation, preprocessing, modeling, evaluation, and visualization.

## Structure
- `data/`: Contains the dataset (`climate_data.csv`).
- `src/`: Source code modules.
  - `data_generator.py`: Generates synthetic climate data.
  - `data_loader.py`: Handles data loading and preprocessing.
  - `models.py`: Contains the `ClimateModel` class (Linear Regression, Random Forest).
  - `visualization.py`: Functions for plotting trends and model performance.
  - `main.py`: The main script to run the pipeline.
- `notebooks/`: Directory for Jupyter notebooks.
- `results/`: Output directory for plots and metrics.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher.

### Installation
1.  **Clone/Open the project** in VS Code.
2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv env
    # Windows:
    .\env\Scripts\activate
    # macOS/Linux:
    source env/bin/activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project
To run the full modeling pipeline, execute the main script:

```bash
python -m src.main
```

This will:
1.  Generate a synthetic dataset in `data/` if it doesn't exist.
2.  Preprocess the data.
3.  Train a Random Forest Regressor.
4.  Evaluate the model and print MAE, MSE, R2 scores.
5.  Generate plots in the `results/` directory (`trends.png`, `prediction_accuracy.png`, `feature_importance.png`).

## Notebooks
You can also run interactive analysis using Jupyter:
```bash
jupyter notebook
```
Navigate to `notebooks/` and create new notebooks or explore the data interactively.

## Model Logic
The synthetic data simulates real-world physics:
- **CO2**: Exponential growth driving temperature.
- **Solar**: Cyclic 11-year patterns.
- **Volcanic**: Random cooling spikes.
- **Sea Level**: Integral of temperature changes.

The model successfully captures these non-linear relationships using Random Forest Regression.
