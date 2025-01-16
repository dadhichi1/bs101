import numpy as np
import pandas as pd
import random
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.stats import norm

# Function to calculate The Greeks
def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:  # put
        delta = -norm.cdf(-d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    return delta, gamma, theta, vega, rho

# Streamlit app setup
st.title("Advanced Options Decision Tool")
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

# Risk-free rate (assumed)
r = 0.05

# Generate option prices (call and put) based on underlying price
call_prices, put_prices = [], []
for idx, price in enumerate(underlying_prices):
    T = (days * intervals_per_day - idx) / (days * intervals_per_day * 365)  # Time to maturity in years
    sigma = np.random.normal(0.2, 0.05)  # Random volatility
    for i in range(-click_range, click_range + 1):
        strike = strike_price + i
        intrinsic_value_call = max(0, price - strike)
        intrinsic_value_put = max(0, strike - price)
        
        # Add extrinsic value (volatility skew and random noise)
        extrinsic_value = max(0.5, np.random.normal(1.5, 0.3))
        call_price = intrinsic_value_call + extrinsic_value
        put_price = intrinsic_value_put + extrinsic_value
        
        delta_c, gamma_c, theta_c, vega_c, rho_c = calculate_greeks(price, strike, T, r, sigma, option_type="call")
        delta_p, gamma_p, theta_p, vega_p, rho_p = calculate_greeks(price, strike, T, r, sigma, option_type="put")
        
        call_prices.append({
            "Underlying": price,
            "Strike": strike,
            "OptionType": "Call",
            "Price": call_price,
            "Delta": delta_c,
            "Gamma": gamma_c,
            "Theta": theta_c,
            "Vega": vega_c,
            "Rho": rho_c,
            "IV": sigma
        })
        put_prices.append({
            "Underlying": price,
            "Strike": strike,
            "OptionType": "Put",
            "Price": put_price,
            "Delta": delta_p,
            "Gamma": gamma_p,
            "Theta": theta_p,
            "Vega": vega_p,
            "Rho": rho_p,
            "IV": sigma
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

# Plot IV vs. historical volatility
st.subheader("Implied Volatility vs. Historical Volatility")
historical_volatility = np.std(np.diff(np.log(underlying_prices))) * np.sqrt(252)  # Annualized historical volatility
iv_data = option_data.groupby('Timestamp')['IV'].mean()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(timestamps[:len(iv_data)], iv_data.values, label="Implied Volatility", color="orange")
ax.axhline(y=historical_volatility, color='blue', linestyle='--', label="Historical Volatility")
ax.set_title("Implied Volatility vs. Historical Volatility")
ax.set_xlabel("Date and Time")
ax.set_ylabel("Volatility")
ax.legend()
st.pyplot(fig)

# Plot Delta trends over time
st.subheader("Delta Trends")
delta_data = option_data.groupby('Timestamp')['Delta'].mean()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(timestamps[:len(delta_data)], delta_data.values, label="Average Delta", color="purple")
ax.set_title("Delta Trends")
ax.set_xlabel("Date and Time")
ax.set_ylabel("Delta")
ax.legend()
st.pyplot(fig)
