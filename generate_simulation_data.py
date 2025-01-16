import numpy as np
import pandas as pd

# Parameters for synthetic data generation
days = 5  # Default to 5 days
intervals_per_day = 12 * 24  # 5-minute intervals in a day
strike_price = 100
click_range = 10

# Generate synthetic underlying price movements (random walk)
np.random.seed(42)
underlying_prices = [strike_price] + list(np.cumsum(np.random.normal(0, 0.5, days * intervals_per_day - 1)) + strike_price)

# Generate option prices (call and put) based on underlying price
call_prices = []
put_prices = []
strikes = range(strike_price - click_range, strike_price + click_range + 1)
for price in underlying_prices:
    extrinsic_value = max(0.5, np.random.normal(1.5, 0.3))
    for strike in strikes:
        intrinsic_value_call = max(0, price - strike)
        intrinsic_value_put = max(0, strike - price)
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

# Save the generated data to a JSON file
option_data.to_json("simulation_data.json", orient="records")

print("Simulation data generated and saved to simulation_data.json")
