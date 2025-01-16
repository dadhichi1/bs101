import streamlit as st
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Streamlit setup
st.title("Nifty 50 Options Trading Simulation")
st.sidebar.header("Simulation Parameters")

# Parameters for simulation
days = st.sidebar.slider("Days", 1, 30, 5)
intervals_per_day = 12 * 6  # 10-minute intervals in a day
strike_price = st.sidebar.number_input("Strike Price", 5000, 20000, 18000)
click_range = st.sidebar.number_input("Click Range", 1, 20, 10)

# Generate synthetic underlying price movements (random walk)
np.random.seed(42)
underlying_prices = [strike_price]
for _ in range(days * intervals_per_day - 1):
    movement = np.random.normal(0, 0.5)
    underlying_prices.append(underlying_prices[-1] + movement)
timestamps = pd.date_range(start='2023-01-01', periods=len(underlying_prices), freq='10T')

# Generate option prices (call and put) based on underlying price
call_prices = []
put_prices = []
for price, timestamp in zip(underlying_prices, timestamps):
    for i in range(-click_range, click_range + 1):
        strike = strike_price + i * 50  # Strike prices in multiples of 50
        intrinsic_value_call = max(0, price - strike)
        intrinsic_value_put = max(0, strike - price)
        
        # Add extrinsic value (volatility skew and random noise)
        extrinsic_value = max(0.5, np.random.normal(1.5, 0.3))
        call_prices.append({
            "Timestamp": timestamp,
            "Underlying": price,
            "Strike": strike,
            "OptionType": "Call",
            "Price": intrinsic_value_call + extrinsic_value
        })
        put_prices.append({
            "Timestamp": timestamp,
            "Underlying": price,
            "Strike": strike,
            "OptionType": "Put",
            "Price": intrinsic_value_put + extrinsic_value
        })

# Combine into a DataFrame
option_data = pd.DataFrame(call_prices + put_prices)

# Simulate 20 random buy/sell decisions over the period
decisions = []
for _ in range(20):
    decision = {
        "Timestamp": random.choice(option_data['Timestamp']),
        "OptionType": random.choice(["Call", "Put"]),
        "Strike": random.choice(range(strike_price - click_range * 50, strike_price + click_range * 50 + 1, 50)),
        "Action": random.choice(["Buy", "Sell"]),
        "Quantity": random.randint(1, 10)
    }
    decisions.append(decision)

decision_df = pd.DataFrame(decisions)

# Classify decisions (Good, Neutral, Bad) based on mean reversion and 100-minute moving average
def classify_decision(row):
    # Retrieve matching option data
    relevant_option = option_data[(option_data['Strike'] == row['Strike']) &
                                  (option_data['OptionType'] == row['OptionType']) &
                                  (option_data['Timestamp'] == row['Timestamp'])]
    if relevant_option.empty:
        return "Neutral", "No matching option data"

    current_price = relevant_option['Price'].values[0]
    historical_prices = option_data.loc[option_data['Timestamp'] <= row['Timestamp'], 'Price']
    mean_price = historical_prices.mean()
    
    # Calculate 100-minute moving average
    moving_avg_period = 100
    if len(historical_prices) < moving_avg_period:
        moving_avg_price = historical_prices.mean()
    else:
        moving_avg_price = historical_prices.iloc[-moving_avg_period:].mean()

    # Evaluate decision based on mean reversion and moving average
    if row['Action'] == "Buy":
        if current_price < mean_price and current_price < moving_avg_price:
            return "Good", "Price below mean and moving average"
        elif current_price == mean_price or current_price == moving_avg_price:
            return "Neutral", "Price equals mean or moving average"
        else:
            return "Bad", "Price above mean and moving average"
    else:
        if current_price > mean_price and current_price > moving_avg_price:
            return "Good", "Price above mean and moving average"
        elif current_price == mean_price or current_price == moving_avg_price:
            return "Neutral", "Price equals mean or moving average"
        else:
            return "Bad", "Price below mean and moving average"

# Apply classification
decision_df[['Classification', 'Note']] = decision_df.apply(
    classify_decision, axis=1, result_type="expand")

# Plot graphs for better understanding
st.subheader("Underlying Price Movement")
fig, ax = plt.subplots()
ax.plot(timestamps, underlying_prices, label="Underlying Price", color="blue")
ax.set_title("Underlying Price Movement")
ax.set_xlabel("Date and Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

st.subheader(f"Option Price Movement for Strike {strike_price}")
call_prices_sample = option_data[(option_data['Strike'] == strike_price) & (option_data['OptionType'] == "Call")]
put_prices_sample = option_data[(option_data['Strike'] == strike_price) & (option_data['OptionType'] == "Put")]

fig, ax = plt.subplots()
ax.plot(call_prices_sample['Timestamp'], call_prices_sample['Price'], label=f"Call Option (Strike={strike_price})", color="green")
ax.plot(put_prices_sample['Timestamp'], put_prices_sample['Price'], label=f"Put Option (Strike={strike_price})", color="red")
ax.set_title(f"Option Price Movement for Strike {strike_price}")
ax.set_xlabel("Date and Time")
ax.set_ylabel("Option Price")
ax.legend()
st.pyplot(fig)

st.subheader("Option Decisions")
st.dataframe(decision_df)

# Display decision classifications
st.subheader("Decision Classifications")
for index, row in decision_df.iterrows():
    st.write(f"Decision {index+1}: {row['Action']} {row['OptionType']} at Strike {row['Strike']}, Classification: {row['Classification']}, Note: {row['Note']}")
