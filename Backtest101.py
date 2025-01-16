
import streamlit as st
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Load pre-generated simulation data
option_data = pd.read_json("simulation_data.json")

# Streamlit setup
st.title("Synthetic Options Data Simulation")
st.sidebar.header("Simulation Parameters")

# Parameters for simulation
strike_price = st.sidebar.number_input("Strike Price", 50, 150, 100, key="strike_price")
click_range = st.sidebar.number_input("Click Range", 1, 20, 10, key="click_range")

# Generate synthetic underlying price movements (random walk)
np.random.seed(42)
days = 5  # Default to 5 days
intervals_per_day = 12 * 24  # 5-minute intervals in a day
underlying_prices = [strike_price] + list(np.cumsum(np.random.normal(0, 0.5, days * intervals_per_day - 1)) + strike_price)

# Simulate 20 random buy/sell decisions over 5 days
decisions = []
strikes = range(strike_price - click_range, strike_price + click_range + 1)
for _ in range(20):
    decision = {
        "Timestamp": random.randint(0, len(underlying_prices) - 1),
        "OptionType": random.choice(["Call", "Put"]),
        "Strike": random.choice(strikes),
        "Action": random.choice(["Buy", "Sell"]),
        "Quantity": random.randint(1, 10)
    }
    decisions.append(decision)

decision_df = pd.DataFrame(decisions)

# Classify decisions (Good, Neutral, Bad) based on mean reversion logic
def classify_decision(row):
    # Retrieve matching option data
    relevant_option = option_data[(option_data['Strike'] == row['Strike']) &
                                  (option_data['OptionType'] == row['OptionType'])]
    if relevant_option.empty:
        return "Neutral", "No matching option data"

    current_price = relevant_option.iloc[row['Timestamp']]['Price']
    lookback_period = 20  # Number of previous intervals to consider for mean reversion
    historical_prices = relevant_option.loc[max(0, row['Timestamp'] - lookback_period):row['Timestamp'], 'Price']
    mean_price = historical_prices.mean()

    # Evaluate decision based on mean reversion
    if row['Action'] == "Buy":
        if current_price < mean_price:  # Buying below recent mean price
            return "Good", "Price below recent mean"
        elif current_price == mean_price:
            return "Neutral", "Price equals recent mean"
        else:
            return "Bad", "Price above recent mean"
    else:  # Sell
        if current_price > mean_price:  # Selling above recent mean price
            return "Good", "Price above recent mean"
        elif current_price == mean_price:
            return "Neutral", "Price equals recent mean"
        else:
            return "Bad", "Price below recent mean"

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
