"""Customer Lifetime Value & Segmentation Engine — Streamlit dashboard.
Place this file beside the exported notebook artifacts and run:
    python -m streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title="Customer Intelligence Dashboard", page_icon="📊", layout="wide")

@st.cache_data
def load_tables():
    scores = pd.read_csv(BASE / "customer_scoring_table.csv", dtype={"Customer ID": str})
    info = pd.read_csv(BASE / "segment_info.csv")
    stake = pd.read_csv(BASE / "revenue_at_stake.csv")
    results = pd.read_csv(BASE / "model_results.csv")
    config = json.loads((BASE / "app_config.json").read_text(encoding="utf-8"))
    scores["Customer ID"] = scores["Customer ID"].str.strip().str.replace(r"\.0$", "", regex=True)
    return scores, info, stake, results, config

@st.cache_resource
def load_models():
    return (joblib.load(BASE / "customer_spending_model.pkl"),
            joblib.load(BASE / "customer_segmentation_pipeline.pkl"))

try:
    scores, info, stake, results, config = load_tables()
except (FileNotFoundError, ValueError, KeyError) as exc:
    st.error(f"Unable to load dashboard data: {exc}. Keep app.py in the same folder as all exported CSV/JSON files.")
    st.stop()

features = config["features"]
rfm_cols = config["rfm_cols"]
cluster_names = info.set_index("Cluster")["Segment"].to_dict()
cluster_actions = info.set_index("Cluster")["Recommended_Action"].to_dict()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Customer Segmentation", "Spending Prediction", "Model Performance"])
st.title("📊 Customer Intelligence Dashboard")
st.caption("Educational demo using historical Online Retail II data (2009–2011). Predictions are estimates, not guarantees. "
           f"Scoring snapshot: {config['scored_as_of'][:10]}. Currency: GBP (£).")

if page == "Overview":
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers scored", f"{len(scores):,}")
    c2.metric("Historical revenue", f"£{scores['Monetary'].sum():,.0f}")
    c3.metric("Predicted next-90-day revenue", f"£{scores['Predicted_Next90d_Spend'].sum():,.0f}")
    st.subheader("Customers by segment")
    st.bar_chart(scores["Segment"].value_counts().rename("Customers"))
    st.subheader("Customer records")
    st.dataframe(scores[["Customer ID", "Recency", "Frequency", "Monetary", "Segment"]],
                 use_container_width=True, hide_index=True)

elif page == "Customer Segmentation":
    st.subheader("Segments and their profiles")
    st.write("Profiles summarize average recency, purchase frequency, and monetary value per segment.")
    profile_cols = ["Segment", "Customers", "Recency", "Frequency", "Monetary", "Recommended_Action"]
    st.dataframe(info[profile_cols].round(2), use_container_width=True, hide_index=True)
    st.subheader("Customers by segment")
    st.bar_chart(info.set_index("Segment")["Customers"])
    st.subheader("Predicted revenue by segment (next 90 days)")
    st.bar_chart(stake.set_index("Segment")["Predicted_90d_Revenue"])
    st.dataframe(stake.round(2), use_container_width=True, hide_index=True)
    st.subheader("Browse customers in a segment")
    segment = st.selectbox("Filter by segment", sorted(scores["Segment"].dropna().unique()))
    subset = scores.loc[scores["Segment"] == segment,
                        ["Customer ID", "Recency", "Frequency", "Monetary", "Segment", "Predicted_Next90d_Spend"]]
    st.dataframe(subset, use_container_width=True, hide_index=True)

elif page == "Spending Prediction":
    st.subheader("Search a customer ID")
    st.write("Look up an existing customer to see their segment, predicted next-90-day spend, and recommended marketing action.")
    cid = st.text_input("Customer ID", placeholder="Example: 12347").strip()
    if cid:
        found = scores.loc[scores["Customer ID"] == cid]
        if found.empty:
            st.warning("Customer ID not found in the exported scoring table. Try an ID shown in the customer records.")
        else:
            customer = found.iloc[0]
            a, b, c = st.columns(3)
            a.metric("Customer ID", str(customer["Customer ID"]))
            b.metric("Segment", str(customer["Segment"]))
            c.metric("Predicted spend — next 90 days", f"£{customer['Predicted_Next90d_Spend']:,.2f}")
            st.success(f"Recommended marketing action: {customer['Recommended_Action']}")
            st.caption("This is the stored model score from the historical scoring snapshot, not a live prediction for today's date.")
            st.subheader("Customer profile")
            st.dataframe(customer[features].to_frame("Value"), use_container_width=True)
    st.divider()
    st.subheader("What-if prediction (optional)")
    st.write("Enter a hypothetical customer's purchase history to get a new model prediction.")
    with st.form("predict_form"):
        left, right = st.columns(2)
        recency = left.number_input("Recency (days)", min_value=0.0, value=30.0)
        frequency = left.number_input("Frequency (orders)", min_value=1.0, value=3.0)
        monetary = left.number_input("Monetary (total spend, £)", min_value=0.0, value=1000.0)
        avg_order = left.number_input("Average order value (£)", min_value=0.0, value=333.33)
        tenure = right.number_input("Customer tenure (days)", min_value=0.0, value=180.0)
        spend90 = right.number_input("Spend in last 90 days (£)", min_value=0.0, value=300.0)
        orders90 = right.number_input("Orders in last 90 days", min_value=0.0, value=1.0)
        submitted = st.form_submit_button("Predict spending")
    if submitted:
        values = dict(zip(["Recency", "Frequency", "Monetary", "AvgOrderValue", "CustomerTenure",
                           "Spend_Last90d", "Orders_Last90d"],
                          [recency, frequency, monetary, avg_order, tenure, spend90, orders90]))
        sample = pd.DataFrame([values])
        try:
            model, segmentation = load_models()
            prediction = max(0.0, float(model.predict(sample[features])[0]))
            cluster = int(segmentation.predict(sample[rfm_cols])[0])
            a, b = st.columns(2)
            a.metric("Predicted spend — next 90 days", f"£{prediction:,.2f}")
            b.metric("Predicted segment", cluster_names.get(cluster, f"Cluster {cluster}"))
            st.info(f"Recommended marketing action: {cluster_actions.get(cluster, 'Not available')}")
        except Exception as exc:
            st.error(f"Prediction could not run: {exc}. Check that both .pkl files are present and dependencies are installed.")

else:
    st.subheader("Initial test-set model comparison")
    st.dataframe(results.round(3), use_container_width=True, hide_index=True)
    if "Test_MAE" in results:
        st.bar_chart(results.set_index("Model")["Test_MAE"])
    st.subheader("Selected model")
    st.write(f"**{config['selected_model']}**")
    a, b, c = st.columns(3)
    a.metric("Test MAE", f"£{config['test_mae']:,.2f}")
    b.metric("Test RMSE", f"£{config['test_rmse']:,.2f}")
    c.metric("Test R²", f"{config['test_r2']:.3f}")
    st.caption("These are the exported notebook test results; the dashboard does not retrain the models.")
