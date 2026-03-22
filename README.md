# AeroPredict-ML: Turbofan Engine Predictive Maintenance ✈️

## Overview
AeroPredict-ML is a machine learning project that predicts the **Remaining Useful Life (RUL)** of turbofan aircraft engines based on time-series sensor data. By transitioning from reactive to predictive maintenance, this model helps identify engine degradation before catastrophic failure occurs.

This project was built using the **NASA C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation) dataset.

## The Aerospace Approach
Instead of blindly feeding data into an AI, this project incorporates physical aerospace principles:
* **Feature Selection:** Filtered out 6 static sensors (like fan bypass duct pressure) that exhibited zero variance over the engine's lifecycle to reduce noise.
* **Piecewise Linear Degradation:** Applied a realistic degradation model capping the healthy RUL at 125 cycles. Engines operate at peak efficiency for most of their life; measurable sensor drift only occurs as wear accelerates near the end of life.

## Model Performance
Using a **Random Forest Regressor** to handle the non-linear, noisy multi-sensor data, the model achieved the following baseline metrics:
* **Mean Absolute Error (MAE):** 13.45 flight cycles
* **Root Mean Squared Error (RMSE):** 18.61 flight cycles
<img width="1009" height="475" alt="results_graph" src="https://github.com/user-attachments/assets/0097b66c-d084-4f0b-8daa-95c8d36f05bc" />

## Project Structure
```text
AeroPredict_ML_Project/
├── assets/                  # Images and graphs
├── data/                    # NASA C-MAPSS .txt datasets
├── notebooks/
│   └── eda_notebook.ipynb   # Exploratory Data Analysis & visual degradation tracking
├── src/
│   └── model_training.py    # Clean production script for model training
├── README.md
└── requirements.txt
