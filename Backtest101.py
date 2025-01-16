import numpy as np
import pandas as pd
import random
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Streamlit app setup
st.title("Options Decision Tool")
st.sidebar.header("Parameters")

# Parameters for synthetic data generation
days = st.sidebar.slider("Number of Days", min_value=1, max_value=30, value=10)
intervals_per_day = st.sidebar.slider("Intervals per Day", min_value=24, max_value=288, value=288)  # 5-min intervals
strike_price = st.sidebar.number_input("Strike Price", value=100)
click_range = st.sidebar.slider("Click Range", min_value=5, max_value=50, value=10)

# Generate synthetic underlying price movements (random walk)
np.random.seed(42)
underlying_prices = np.cumsum(np.random.normal(0, 0.5, days * intervals_per_day)) + strike_price

# Generate timestamps for each interval
start_time = datetime(2025, 1, 1, 9, 15)  # NIFTY50 trading start time
timestamps = [start_time + timedelta(minutes=5 * i) for i in range(days * intervals_per_day)]

# Generate option prices (call and put) based on underlying price
call_prices, put_prices = [], []
for price in underlying_prices:
    for i in range(-click_range, click_range + 1):
        strike = strike_price + i
        intrinsic_value_call = max(0, price - strike)
        intrinsic_value_put = max(0, strike - price)
        
        # Add extrinsic value (volatility skew and random noise)
        extrinsic_value = max(0.5, np.random.normal(1.5, 0.3))
        call_prices.append({
            "Underlying": price,
            "Strike": strike,
            "OptionType": "Call",
            "Price": intrinsic_value_call + extrinsic_value
        })
        put_prices.append({
            "Underlying": price,
            "Strike": strike,
            "OptionType": "Put",
            "Price": intrinsic_value_put + extrinsic_value
        })

# Combine into a DataFrame
option_data = pd.DataFrame(call_prices + put_prices)

# Simulate random buy/sell decisions
num_decisions = st.sidebar.slider("Number of Decisions", min_value=5, max_value=100, value=20)
decisions = [{
    "Timestamp": random.randint(0, len(underlying_prices) - 1),
    "OptionType": random.choice(["Call", "Put"]),
    "Strike": random.choice(range(strike_price - click_range, strike_price + click_range + 1)),
    "Action": random.choice(["Buy", "Sell"]),
    "Quantity": random.randint(1, 10)
} for _ in range(num_decisions)]

decision_df = pd.DataFrame(decisions)

# Classify decisions (Good, Neutral, Bad) based on popular quant strategies
def classify_decision(row):
    relevant_option = option_data[(option_data['Strike'] == row['Strike']) &
                                  (option_data['OptionType'] == row['OptionType'])]
    if relevant_option.empty:
        return "Neutral", "No matching option data"

    current_price = relevant_option.iloc[row['Timestamp']]['Price']
    historical_prices = relevant_option.loc[:row['Timestamp'], 'Price']
    mean_price = historical_prices.mean()

    # Based on mean reversion strategy
    if row['Action'] == "Buy":
        if current_price < mean_price:
            return "Good", "Price below historical average - Mean Reversion Buy"
        elif current_price == mean_price:
            return "Neutral", "Price equals historical average"
        else:
            return "Bad", "Price above historical average - Mean Reversion Sell"
    else:
        if current_price > mean_price:
            return "Good", "Price above historical average - Mean Reversion Sell"
        elif current_price == mean_price:
            return "Neutral", "Price equals historical average"
        else:
            return "Bad", "Price below historical average - Mean Reversion Buy"

# Apply classification
decision_df[['Classification', 'Note']] = decision_df.apply(
    classify_decision, axis=1, result_type="expand")

# Display decision table
st.subheader("Option Decisions")
st.dataframe(decision_df)

# Plot underlying price trend
st.subheader("Underlying Price Movement")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(timestamps, underlying_prices, label="Underlying Price", color="blue")
ax.set_title("Underlying Price Movement")
ax.set_xlabel("Date and Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Plot option price trends for a sample strike
sample_strike = st.sidebar.number_input("Sample Strike for Graph", value=strike_price)
call_prices_sample = option_data[(option_data['Strike'] == sample_strike) & (option_data['OptionType'] == "Call")]
put_prices_sample = option_data[(option_data['Strike'] == sample_strike) & (option_data['OptionType'] == "Put")]

st.subheader(f"Option Price Movement for Strike {sample_strike}")
fig, ax = plt.subplots(figsize=(12, 6))
if not call_prices_sample.empty:
    ax.plot(timestamps[:len(call_prices_sample)], call_prices_sample['Price'].values, label=f"Call Option (Strike={sample_strike})", color="green")
if not put_prices_sample.empty:
    ax.plot(timestamps[:len(put_prices_sample)], put_prices_sample['Price'].values, label=f"Put Option (Strike={sample_strike})", color="red")
ax.set_title(f"Option Price Movement for Strike {sample_strike}")
ax.set_xlabel("Date and Time")
ax.set_ylabel("Option Price")
ax.legend()
st.pyplot(fig)
