# NYC Flash-Freeze Risk Prediction

This project builds a machine learning model to predict next-day flash-freeze risk in New York City using historical weather data.

## Problem

Flash-freeze events occur when precipitation is followed by a rapid drop below freezing temperatures, creating hazardous icy conditions.  
The goal is to predict the probability of a flash-freeze event **one day in advance**.

## Data

Daily historical weather data for NYC (Central Park station) from 2010–2025 was used, including:

- Maximum temperature
- Minimum temperature
- Average temperature
- Precipitation
- Snowfall

Additional engineered features include:

- Yesterday’s weather values
- 3-day rolling precipitation and temperature averages
- Temperature swing (daily max − min)
- Seasonal indicators

The dataset is restricted to winter months (November–March).

## Label Definition

A flash-freeze event is defined as:

- Precipitation on day *t*, and  
- Maximum temperature above 1°C on day *t+1*, and  
- Minimum temperature below −1°C on day *t+1*

The model predicts whether this event will occur the following day.

## Modeling Approach

- Model: XGBoost classifier  
- Time-based train/test split (no data leakage)  
- Class imbalance handled using `scale_pos_weight`  
- Threshold tuned using F1 score for rare-event detection  

## Evaluation

Metrics reported:

- ROC-AUC
- Precision–Recall AUC
- Classification report at tuned threshold

The model achieves strong ranking performance (ROC-AUC ~0.87) on a holdout test period.

## Visualizations

The following plots are generated and saved in the `reports/` directory:

- Precision–Recall curve  
- Risk probability over time with actual flash-freeze events marked  
- SHAP feature importance summary  

## Outputs

Saved artifacts:

- Trained model (`models/xgb_flash_freeze.joblib`)
- SHAP summary plot
- Precision–Recall curve
- Risk-over-time plot

## How to Run

```bash
python3 src/build_dataset.py
python3 src/train.py
python3 src/predict.py
```
