import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, IsolationForest

# Page setup
st.set_page_config(page_title="Grocery Demand Forecasting", layout="centered")

# Simple UI touch
st.markdown("""
<style>
.stApp {
    background-color: #fafafa;
}
h1 {
    color: #222;
}
</style>
""", unsafe_allow_html=True)

st.title("🛒 Grocery Demand Forecasting")

# Load data
df = pd.read_csv("grocery_chain_data.csv")
df.columns = df.columns.str.lower()
df["transaction_date"] = pd.to_datetime(df["transaction_date"])

# Create daily sales data
daily = (
    df.groupby("transaction_date")["final_amount"]
    .sum()
    .reset_index()
)

daily.columns = ["Date", "Sales"]
daily = daily.sort_values("Date")

# Features for the model
daily["day"] = daily["Date"].dt.day
daily["month"] = daily["Date"].dt.month
daily["weekday"] = daily["Date"].dt.weekday

daily["lag_1"] = daily["Sales"].shift(1)
daily.dropna(inplace=True)

# Train demand model
X = daily[["day", "month", "weekday", "lag_1"]]
y = daily["Sales"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Detect unusual sales days
iso = IsolationForest(contamination=0.05, random_state=42)
daily["anomaly"] = iso.fit_predict(daily[["Sales"]])

# Smooth sales for better visualization
daily["Sales_Smoothed"] = daily["Sales"].rolling(7).mean()
anomalies = daily[daily["anomaly"] == -1]

# Plot results
plt.figure(figsize=(12, 5))

plt.plot(
    daily["Date"],
    daily["Sales_Smoothed"],
    color="blue",
    linewidth=2,
    label="Sales trend"
)

plt.scatter(
    anomalies["Date"],
    anomalies["Sales"],
    color="red",
    s=40,
    label="Unusual sales"
)

plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(alpha=0.2)

st.pyplot(plt)

st.markdown(
    "Blue line shows the overall sales trend. Red dots mark unusual sales days."
)


with st.expander("🔍 View detected anomalies"):
    st.dataframe(anomalies[["Date", "Sales"]])
