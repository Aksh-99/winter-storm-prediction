# Flash-Freeze Risk Prediction – NYC

## Overview

This project builds a machine learning system to predict flash-freeze events in New York City 24 hours in advance using historical weather data.

A flash-freeze occurs when precipitation is followed by a rapid temperature drop below freezing, creating dangerous ice conditions on roads and sidewalks.

This is a rare-event classification problem focused on risk prediction rather than general weather forecasting.

---

## Why This Matters (Impact)

Flash-freeze events are particularly hazardous because they:

- Create black ice with little visible warning  
- Increase vehicle accidents and pedestrian injuries  
- Disrupt morning commute and public transit  
- Require rapid municipal salt deployment  

Accurately predicting flash-freeze risk in advance can help:

- City services pre-treat roads  
- Transit systems prepare for delays  
- Schools and businesses make informed decisions  
- Infrastructure planners reduce winter-related hazards  

This model predicts flash-freeze risk 24 hours in advance with approximately 0.87 ROC-AUC, improving rare-event recall by roughly 13% over a logistic regression baseline.  
Such a system could support data-driven winter response planning and reduce avoidable slip-related incidents.

---

## Data

Source:  
National Oceanic and Atmospheric Administration (NOAA) historical weather data.

Features include:

- Temperature (current and rolling averages)  
- Precipitation  
- Wind speed  
- Atmospheric pressure  
- Humidity  
- Temperature change over 6–12 hours  
- Seasonal indicators (winter flag, month)  

Target variable:

Binary classification:
- 1 = Flash-freeze event  
- 0 = No flash-freeze  

Flash-freeze is defined as:
- Precipitation occurs  
- Temperature drops to ≤ 32°F within a defined time window  

---

## Modeling Approach

Baseline model:
- Logistic Regression  

Final model:
- XGBoost Classifier  

XGBoost was selected because it:

- Captures nonlinear feature interactions  
- Performs strongly on structured tabular data  
- Handles imbalanced classification effectively  

Evaluation strategy:

- Time-based train/test split to prevent leakage  
- Cross-validation  
- Metrics:
  - ROC-AUC  
  - Precision-Recall AUC  
  - F1 Score  
  - Confusion Matrix  

---

## Results

| Model                | ROC-AUC | PR-AUC | Recall |
|----------------------|---------|--------|--------|
| Logistic Regression  | 0.78    | 0.42   | 0.61   |
| XGBoost              | 0.87    | 0.58   | 0.74   |

Key improvements:

- +9% ROC-AUC over baseline  
- +13% improvement in recall for rare flash-freeze events  
- Stronger identification of high-risk days while controlling false positives  

The model demonstrates improved discrimination and better detection of hazardous conditions compared to a linear baseline.

---

## Key Insights (Interpretability)

Using SHAP analysis, the most important predictors include:

- Rapid temperature drop (6–12 hour delta)  
- Precipitation intensity  
- Wind speed  
- Late-evening timing  
- Prior-day ground temperature  

Notable finding:

Precipitation alone is not sufficient to predict flash-freeze risk.  
The rate of temperature decline is the strongest driver.

This confirms domain intuition while quantifying actionable risk thresholds.

---

## Real-World Usefulness (Decision Support)

This system can support:

- Municipal winter response planning  
- Infrastructure preparedness  
- Transit reliability management  
- Insurance and climate risk modeling  

The modeling framework also generalizes to other rare-event prediction domains such as credit risk, fraud detection, and operational disruption forecasting.

---

## Repository Structure
```
data/ # Raw and processed datasets
notebooks/ # Exploratory analysis and modeling notebooks
src/ # Feature engineering and training scripts
models/ # Saved trained models
```
---

## Conclusion

This project demonstrates:

- Clear problem framing around rare-event risk  
- Time-series feature engineering  
- Model comparison against a baseline  
- Proper evaluation for imbalanced data  
- Interpretability using SHAP  
- Practical decision-support orientation  

Rather than simply predicting snowfall, this system focuses on actionable winter hazard risk, making it relevant to infrastructure planning, urban safety, and applied risk analytics.
