# Winter Storm Risk Prediction System  
### A Data-Driven Infrastructure Risk Modeling Project for New York City

## 1. Executive Summary

Severe winter storms in New York City create significant public safety risks, transportation disruptions, and infrastructure strain. Flash-freeze events in particular are difficult to anticipate and can lead to hazardous road conditions, transit delays, and emergency response challenges.

This project builds a machine learning–based winter storm risk prediction system using historical meteorological data. The system predicts the probability of flash-freeze events and provides interpretable insights into the atmospheric conditions that drive high-risk scenarios.

The goal is not just to classify weather events, but to design a reproducible and extensible risk modeling pipeline that could support proactive infrastructure planning and operational decision-making.

---

## 2. Problem Statement

Flash-freeze events occur when temperatures rapidly drop below freezing following precipitation. These events:

- Increase accident rates
- Disrupt transportation systems
- Require rapid municipal response
- Create public safety hazards

Traditional weather forecasts provide raw temperature and precipitation values but do not directly quantify flash-freeze risk.

This project reframes the problem as:

> Can we predict the probability of a flash-freeze event using historical weather data and machine learning?

---

## 3. Dataset

### Source
Historical daily weather data from a NYC weather station (multi-year dataset).

### Observations
- 50,000+ daily records
- Multiple meteorological features including:
  - Temperature (min, max, mean)
  - Precipitation
  - Wind speed
  - Atmospheric pressure
  - Humidity

### Target Variable
Binary indicator:
- 1 = Flash-freeze event
- 0 = No event

The target was engineered using domain logic combining precipitation and temperature thresholds.

---

## 4. Methodology

### 4.1 Data Processing

- Cleaned missing values
- Engineered temporal features
- Created derived variables capturing temperature deltas and precipitation interactions
- Ensured no target leakage during feature generation

### 4.2 Modeling Approach

Primary Model:
- XGBoost Classifier

Baseline Comparison:
- Logistic Regression

Evaluation Strategy:
- Train/test split
- ROC-AUC as primary metric
- Precision-Recall analysis
- Confusion matrix

### 4.3 Model Performance

- ROC-AUC: ~0.87
- Significant improvement over baseline logistic regression
- Balanced precision and recall

The model demonstrates strong discriminative ability in identifying high-risk flash-freeze conditions.

---

## 5. Model Interpretability

To ensure transparency and explainability:

- SHAP (SHapley Additive Explanations) was used
- Feature importance analysis conducted
- Global and local explanation patterns examined

Key Drivers Identified:
- Rapid temperature drops
- Precipitation intensity
- Low minimum temperatures
- Pressure shifts

This interpretability layer enables the model to function not just as a predictor, but as an analytical tool for understanding risk drivers.

---

## 6. System Architecture

The project is structured as a modular pipeline:

1. Data ingestion
2. Feature engineering
3. Model training
4. Model evaluation
5. Prediction script

Repository Structure:

- `data_building.py` — Data preprocessing and feature generation
- `train.py` — Model training and evaluation
- `predict.py` — Inference pipeline
- `requirements.txt` — Dependencies

The pipeline design allows reproducibility and future extensibility.

---

## 7. Business and Infrastructure Impact

This system could support:

- Proactive road salting and de-icing
- Transit schedule adjustment
- Emergency resource allocation
- Risk-based municipal alerts

Rather than reacting to weather forecasts, decision-makers could operate on quantified risk probabilities.

Potential Extensions:
- Cost-sensitive modeling to penalize false negatives
- Multi-day forward risk prediction
- Multi-station spatial modeling
- Deployment as an API or dashboard

---

## 8. Limitations

- Single-station dataset
- Binary classification rather than severity scoring
- No real-time deployment
- Limited spatial generalization

These constraints are acknowledged as areas for future enhancement.

---

## 9. Future Roadmap

Planned Improvements:

- Multi-source data integration (NYC Open Data, transit data)
- Ensemble model comparison
- Time-series modeling (LSTM or temporal boosting)
- Deployment via Streamlit or REST API
- Infrastructure cost impact modeling

The long-term goal is to evolve this from a classification model into a full winter storm risk intelligence system.

---

## 10. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib / Seaborn

---

## 11. Reproducibility

To run the project locally:

```bash
pip install -r requirements.txt
python data_building.py
python train.py
python predict.py
```
