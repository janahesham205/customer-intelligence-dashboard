# Customer Lifetime Value & Segmentation Engine

A machine-learning project using historical **UCI Online Retail II** transactions to segment customers and estimate their **spend over the next 90 days**. The accompanying Streamlit dashboard presents segment profiles, customer-level predictions, and suggested marketing actions.

**[Live Streamlit dashboard](https://customer-intelligence-dashboard-j4zwefcqdmfpnmft5bsmnp.streamlit.app/)**

## Business problem
Retail teams need to prioritize retention and marketing effort. This project combines **RFM (Recency, Frequency, Monetary)** segmentation with customer-level spend forecasting to help identify valuable, at-risk, and inactive customers.

## Dataset and methodology
- Source: UCI Online Retail II, historical transactions from 2009–2011.
- Data preparation: cleaning, customer-level aggregation, time-based feature engineering, and leakage-aware modeling.
- Features: Recency, Frequency, Monetary, average order value, customer tenure, spend in the previous 90 days, and orders in the previous 90 days.
- Segmentation: K-Means and hierarchical clustering were explored; the exported app uses a fitted segmentation pipeline.
- Prediction: mean, median, and last-90-day baselines compared with Ridge, Random Forest, and Gradient Boosting, including log-target variants.
- Evaluation: cross-validation and hyperparameter tuning on training data, followed by a held-out test evaluation, residual/error analysis, and feature importance.

## Exported model results
| Model | Test MAE (£) | Test RMSE (£) | Test R² |
|---|---:|---:|---:|
| Gradient Boosting (log target) | 378.91 | 1209.70 | 0.631 |
| **Random Forest (log target) — deployed model** | **392.79** | **1451.34** | **0.469** |
| Last-90-day spend baseline | 399.47 | 1630.69 | 0.330 |

**Important:** Gradient Boosting (log target) has better held-out MAE, RMSE, and R² than the exported Random Forest. The deployed artifact is nevertheless Random Forest (log target), as recorded in `app_config.json`. The reason for that selection should be checked against the notebook's selection criteria before claiming it is the overall best model.

## Dashboard features
1. **Overview:** scored customers, historical revenue, predicted next-90-day revenue, and segment counts.
2. **Customer Segmentation:** segment profiles, predicted revenue by segment, and customer filtering.
3. **Spending Prediction:** look up a customer ID to see segment, stored 90-day spend prediction, and marketing recommendation; optional what-if prediction form.
4. **Model Performance:** held-out model comparisons and selected-model metrics.

The scoring export contains **5,852 customers**. Its predicted next-90-day revenue sums to approximately **£1.62 million**. These figures describe a **historical scoring snapshot (2011-12-09)**, not current retail revenue.

## Run locally
Requires Python and the dependencies in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Keep `app.py`, `app_config.json`, the CSV exports, and both `.pkl` model files in the same directory. Example lookup: **Customer ID `12347`** (if using the provided scoring export).

## Repository contents
- `app.py` — Streamlit user interface
- `app_config.json` — feature list, scoring dates, selected model, and metrics
- `customer_scoring_table.csv` — historical customer-level scores and recommended actions
- `segment_info.csv`, `revenue_at_stake.csv`, `model_results.csv` — dashboard summaries
- `customer_spending_model.pkl`, `customer_segmentation_pipeline.pkl` — fitted model artifacts
- `requirements.txt` — application dependencies
- `Customer_Lifetime_Value_Final_Project.ipynb` — **add the original notebook to this repository for reproducibility**

## Limitations and responsible use
- Historical data and customer behavior may not represent today's market.
- Predicted revenue is an estimate, not guaranteed spending or causal campaign uplift.
- Customer IDs and purchase histories should be reviewed for privacy and publishing permissions before public deployment.
- The what-if form should be used with realistic, internally consistent feature values.
- Re-train and validate models on newer data before operational use.

## Suggested future improvements
Evaluate the selection criterion behind the deployed model, compare against the stronger Gradient Boosting result, monitor drift, and test marketing actions through controlled experiments.
