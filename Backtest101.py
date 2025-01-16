import streamlit as st
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Streamlit setup
st.title("Synthetic Options Data Simulation")
st.sidebar.header("Simulation Parameters")

# Parameters for synthetic data generation
days = st.sidebar.slider("Days", 1, 30, 10)
intervals_per_day = 12 * 24  # 5-minute intervals in a day
strike_price = st.sidebar.number_input("Strike Price", 50, 150, 100)
click_range = st.sidebar.number_input("Click Range", 1, 20, 10)

# Generate synthetic underlying price movements (random walk)
np.random.seed(42)
underlying_prices = [strike_price]
for _ in range(days * intervals_per_day - 1):
    movement = np.random.normal(0, 0.5)
    underlying_prices.append(underlying_prices[-1] + movement)

# Generate option prices (call and put) based on underlying price
call_prices = []
put_prices = []
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

# Simulate 20 random buy/sell decisions over 10 days
decisions = []
for _ in range(20):
    decision = {
        "Timestamp": random.randint(0, len(underlying_prices) - 1),
        "OptionType": random.choice(["Call", "Put"]),
        "Strike": random.choice(range(strike_price - click_range, strike_price + click_range + 1)),
        "Action": random.choice(["Buy", "Sell"]),
        "Quantity": random.randint(1, 10)
    }
    decisions.append(decision)

decision_df = pd.DataFrame(decisions)

# Classify decisions (Good, Neutral, Bad) based on historical data
def classify_decision(row):
    # Retrieve matching option data
    relevant_option = option_data[(option_data['Strike'] == row['Strike']) &
                                   (option_data['OptionType'] == row['OptionType'])]
    if relevant_option.empty:
        return "Neutral", "No matching option data"

    current_price = relevant_option.iloc[row['Timestamp']]['Price']
    historical_prices = relevant_option.loc[:row['Timestamp'], 'Price']
    mean_price = historical_prices.mean()

    # Evaluate decision based on historical mean price
    if row['Action'] == "Buy":
        if current_price < mean_price:  # Buying below historical average price
            return "Good", "Price below historical average"
        elif current_price == mean_price:
            return "Neutral", "Price equals historical average"
        else:
            return "Bad", "Price above historical average"
    else:  # Sell
        if current_price > mean_price:  # Selling above historical average price
            return "Good", "Price above historical average"
        elif current_price == mean_price:
            return "Neutral", "Price equals historical average"
        else:
            return "Bad", "Price below historical average"

# Apply classification
decision_df[['Classification', 'Note']] = decision_df.apply(
    classify_decision, axis=1, result_type="expand")

# Plot graphs for better understanding
st.subheader("Underlying Price Movement")
fig, ax = plt.subplots()
ax.plot(underlying_prices, label="Underlying Price", color="blue")
ax.set_title("Underlying Price Movement")
ax.set_xlabel("Time (5-min intervals)")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

st.subheader(f"Option Price Movement for Strike {strike_price}")
call_prices_sample = option_data[(option_data['Strike'] == strike_price) & (option_data['OptionType'] == "Call")]
put_prices_sample = option_data[(option_data['Strike'] == strike_price) & (option_data['OptionType'] == "Put")]

fig, ax = plt.subplots()
ax.plot(call_prices_sample['Price'].values, label=f"Call Option (Strike={strike_price})", color="green")
ax.plot(put_prices_sample['Price'].values, label=f"Put Option (Strike={strike_price})", color="red")
ax.set_title(f"Option Price Movement for Strike {strike_price}")
ax.set_xlabel("Time (5-min intervals)")
ax.set_ylabel("Option Price")
ax.legend()
st.pyplot(fig)

st.subheader("Option Decisions")
st.dataframe(decision_df)

# Display decision classifications
st.subheader("Decision Classifications")
for index, row in decision_df.iterrows():
    st.write(f"Decision {index+1}: {row['Action']} {row['OptionType']} at Strike {row['Strike']}, Classification: {row['Classification']}, Note: {row['Note']}")
