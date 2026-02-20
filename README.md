# NYC Flash-Freeze Risk Prediction

## Overview

This project builds a machine learning system to predict next-day flash-freeze risk in New York City using historical meteorological data.

A flash-freeze event occurs when precipitation is followed by a rapid drop below freezing temperatures, creating hazardous icy conditions. The objective is to estimate the probability of such events occurring the following day.

The system includes:

- Data ingestion and feature engineering
- Time-based model training and evaluation
- Rare-event threshold optimization
- Model interpretability using SHAP
- Risk visualization and performance analysis

---

## Problem Definition

Goal: Predict the probability of a flash-freeze event occurring tomorrow using today's and recent weather conditions.

Flash-freeze proxy label:

- Precipitation today > 0
- Tomorrow's max temperature > 1°C
- Tomorrow's min temperature < -1°C

This formulation ensures the model predicts future risk rather than using same-day information (avoiding data leakage).

---

## Data

Source: Historical daily weather data for NYC (Central Park station).

Time Range: 2010–2025 (winter months only: November–March)

Key Variables:

- Temperature (max, min, average)
- Precipitation
- Snowfall
- Temperature range
- Lagged weather variables
- Rolling 3-day aggregates
- Seasonal indicators

Total observations: 2,420 winter days  
Positive rate (flash-freeze days): ~11.8%

---

## Feature Engineering

Engineered features include:

- Yesterday's precipitation and snowfall
- 3-day rolling precipitation and snowfall totals
- Temperature swing (daily max − min)
- 3-day rolling temperature averages
- Seasonal encodings (day-of-year sine/cosine)
- Day-of-week

The model uses lagged and rolling features to capture short-term weather dynamics.

---

## Model

Algorithm: XGBoost Classifier

Why XGBoost:
- Handles nonlinear relationships
- Performs well on tabular data
- Supports class imbalance weighting

Class imbalance handling:
- scale_pos_weight applied (~7.4) due to ~12% positive rate

Train/Test Split:
- Time-based split at 2023-01-01
- No random shuffling (prevents temporal leakage)

---

## Evaluation

### Metrics

- ROC-AUC: ~0.87
- PR-AUC: ~0.38 (more appropriate for rare events)
- Positive class rate: ~11.8%

PR-AUC significantly exceeds baseline (~0.12), indicating meaningful signal.

### Threshold Optimization

Two operating modes were evaluated:

1. Default threshold (0.50)
   - Moderate recall (~38%)
   - Fewer false alarms

2. F1-optimized threshold (~0.075)
   - High recall (~92%)
   - Increased false positives

This demonstrates the tradeoff between safety-first detection and alert fatigue.

---

## Visualizations

Saved in the `reports/` directory:

1. Precision–Recall Curve  
   - Shows performance for rare-event detection  
   - Includes baseline positive rate reference  

2. Risk Probability Over Time  
   - Predicted risk plotted across the test period  
   - True flash-freeze days marked  
   - Alert threshold displayed  

3. SHAP Feature Importance  
   - Identifies key drivers of flash-freeze risk  
   - Uses human-readable feature names  

---

## Model Interpretability

SHAP analysis shows that flash-freeze risk is primarily influenced by:

- Recent precipitation
- Temperature swings
- Recent average temperature
- Seasonal timing

This aligns with physical intuition and increases model trust.

---

## Project Structure
