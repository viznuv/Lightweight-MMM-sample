# Lightweight Marketing Mix Model (LightweightMMM)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a simplified, lightweight implementation of a Marketing Mix Model (MMM) in Python.  It's designed for educational purposes, smaller datasets, and faster iteration compared to more complex, production-ready MMM solutions.  This model focuses on the core principles of MMM: adstock, saturation, and optimization.

**Key Features:**

*   **Adstock:** Implements a geometric adstock transformation to account for the lagged effect of advertising.  Supports both optimized adstock rate (per channel) and a fixed rate.
*   **Saturation:** Includes both Hill and Exponential saturation functions to model diminishing returns on media spend.  The Hill function is generally recommended for its flexibility.
*   **Optimization:** Uses `scipy.optimize.minimize` to find the optimal parameters (coefficients, adstock rates, saturation parameters) that minimize the Root Mean Squared Error (RMSE) between predicted and actual sales.
*   **Channel Contributions:** Provides a method to calculate the contribution of each media channel to the overall predicted sales.
*   **Budget Allocation (Example):** Includes a basic example demonstrating how to use the fitted model for budget optimization, finding the allocation that maximizes predicted sales given a total budget constraint.
*   **Clear Visualization:** Includes functions to plot actual vs. predicted sales and model residuals.
*   **Easy to Use:**  The `LightweightMMM` class provides a simple interface for initializing, training, and analyzing the model.
*   **Self-Contained Example:** The provided code includes data generation, making it easy to run and experiment with immediately.

**Why Lightweight?**

This implementation is considered "lightweight" because it intentionally omits features commonly found in production-level MMMs, such as:

*   **Trend and Seasonality:**  No built-in handling of time-based trends or seasonal patterns.  You would need to pre-process your data or add these features manually.
*   **Hierarchical Priors (Bayesian Methods):**  This is a frequentist model.  It does not use Bayesian techniques or hierarchical priors, which are common in more robust MMM libraries like Robyn.
*   **External Regressors:** No direct support for incorporating external factors (e.g., economic indicators, competitor activity).  These would need to be added as additional columns in your input data.
*   **Complex Interactions:**  The model assumes additive effects of media channels.  It doesn't model complex interactions between channels.

These simplifications make the code easier to understand and modify, and it runs much faster than Bayesian MMMs. It's ideal for learning the basics of MMM or for situations where a simpler model is sufficient.

**Installation:**

The required libraries are:

*   pandas
*   numpy
*   matplotlib
*   scipy

You can install them using pip:

```bash
pip install pandas numpy matplotlib scipy
Usage:

```python
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import 1  random  # Import the random module   
1.
github.com
github.com

--- Data Preparation (Example - Replace with your data) ---
num_weeks = 104
dates = pd.date_range(start="2021-01-01", periods=num_weeks, freq="W")
sales = 1000 + 50 * np.random.randn(num_weeks)  # Base sales + noise
tv_spend = 500 + 200 * np.random.rand(num_weeks)
digital_spend = 300 + 150 * np.random.rand(num_weeks)
search_spend = 100 + 50 * np.random.rand(num_weeks)

data = pd.DataFrame({
'date': dates,
'sales': sales,
'TV': tv_spend,
'Digital': digital_spend,
'Search': search_spend
})

--- Model Initialization and Training ---
from lightweight_mmm import LightweightMMM  # Assuming your code is in lightweight_mmm.py

model = LightweightMMM(
data=data,
media_cols=['TV', 'Digital', 'Search'],
adstock_max_lag=8,
saturation_method='hill',
hill_k_bounds=(0.1, 3),
hill_s_bounds=(0.5, 2),
optimize_adstock=True,
random_seed=42
)

--- Results and Analysis ---
print("\nOptimized Parameters:")
print(model.params)

print(f"\nRMSE: {model.calculate_rmse():.2f}")
print(f"R-squared: {model.calculate_r_squared():.4f}")

model.plot_predictions()
model.plot_residuals()

contributions = model.get_channel_contributions()
print("\nChannel Contributions (first 5 rows):")
print(contributions.head())

--- Budget Optimization (Example) ---
def simulate_sales(budget_allocation, model):
# ... (same simulate_sales function as in the code) ...
simulated_data = model.data.copy()
total_budget = sum(budget_allocation)
# Check if budget allocation is valid
if len(budget_allocation) != len(model.media_cols):
raise ValueError("Length of budget allocation must match the number of media channels.")
if total_budget <= 0:
raise ValueError("Total budget must be greater than zero.")
if any(b < 0 for b in budget_allocation):
raise ValueError("Budget allocation values cannot be negative.")
# Apply the new budget
for i, channel in enumerate(model.media_cols):
simulated_data.loc[simulated_data.index[-1], channel] = budget_allocation[i]
# parameters array
params_array = [model.params['intercept']]
for ch in model.media_cols:
params_array.append(model.params[ch]['beta'])
if model.optimize_adstock:
params_array.append(model.params[ch]['adstock_rate'])
if model.saturation_method == 'hill':
params_array.append(model.params[ch]['hill_k'])
params_array.append(model.params[ch]['hill_s'])
else:
params_array.append(model.params[ch]['exp_k'])

predicted_sales = model._mmm_equation(np.array(params_array), simulated_data)
return -predicted_sales[-1]
constraints
total_budget = data[['TV', 'Digital', 'Search']].iloc[-1].sum()
constraints = ({'type': 'eq', 'fun': lambda x:  sum(x) - total_budget})
bounds = [(0, total_budget) for _ in range(len(model.media_cols))]

initial guess
x0 = [total_budget / len(model.media_cols)] * len(model.media_cols)

optimization
result = minimize(simulate_sales, x0, args=(model,), method='SLSQP', bounds=bounds, constraints=constraints)
if result.success:
optimal_allocation = result.x
print("\nOptimal Budget Allocation:")
for i, channel in enumerate(model.media_cols):
print(f"{channel}: {optimal_allocation[i]:.2f}")
print(f"Projected Sales: {-result.fun:.2f}")
equal_allocation = [total_budget / len(model.media_cols)] * len(model.media_cols)
equal_sales = -simulate_sales(equal_allocation, model)
print("\nEqual Budget Allocation:")
for i, channel in enumerate(model.media_cols):
print(f"{channel}: {equal_allocation[i]:.2f}")
print(f"Projected Sales (Equal Allocation): {equal_sales:.2f}")
else:
print("\nBudget optimization failed.")
print(result.message)


**Class `LightweightMMM`**

  * **`__init__(self, data, media_cols, adstock_max_lag=8, saturation_method='hill', hill_k_bounds=(0.1, 5), hill_s_bounds=(0.1, 3), optimize_adstock=True, adstock_rate=0.7, random_seed=42)`:**

      * `data`: Input DataFrame. Must contain 'date', 'sales', and media channel columns.
      * `media_cols`: List of media channel column names.
      * `adstock_max_lag`: Maximum lag for adstock (in periods, e.g., weeks).
      * `saturation_method`: 'hill' (recommended) or 'exponential'.
      * `hill_k_bounds`: Bounds for the 'k' parameter in the Hill function.
      * `hill_s_bounds`: Bounds for the 's' parameter in the Hill function.
      * `optimize_adstock`: Whether to optimize adstock rates.
      * `adstock_rate`: Fixed adstock rate if `optimize_adstock` is False.
      * `random_seed`: Random seed for reproducibility.

  * **`_adstock(self, x, rate, max_lag)`:** Calculates the adstock transformation.

  * **`_saturation(self, x, k, s, method='hill')`:** Applies the saturation function.

  * **`_mmm_equation(self, params_array, data)`:** The core MMM equation.

  * **`_objective_function(self, params_array, data)`:** Calculates the RMSE.

  * **`_optimize(self)`:** Performs the parameter optimization.

  * **`get_channel_contributions(self)`:** Calculates channel contributions.

  * **`plot_predictions(self)`:** Plots actual vs. predicted sales.

  * **`plot_residuals(self)`:** Plots residuals.

  * **`calculate_rmse(self)`:** Returns the RMSE

  * **`calculate_r_squared(self)`:** Returns R-squared

**Function `simulate_sales`**

  * **`simulate_sales(budget_allocation, model)`:**  This function is used for the budget optimization example.  It takes a proposed `budget_allocation` (a list of spend amounts for each channel) and the fitted `model` object.  It modifies the *last row* of the input data to reflect the new budget allocation and then uses the model's parameters to predict the resulting sales.  The negative of the predicted sales is returned because we are using a minimization function to *maximize* sales.

**Budget Optimization Example**

The example demonstrates how to use `scipy.optimize.minimize` with the `simulate_sales` function to find an optimal budget allocation.  It uses the `SLSQP` solver with constraints (total budget) and bounds (non-negative spend).

**Further Development:**

  * **Add Trend and Seasonality:** Implement components to capture trend and seasonal effects, either through data preprocessing or by adding terms to the `_mmm_equation`.
  * **External Regressors:** Allow the inclusion of external variables (e.g., economic data) as additional predictors.
  * **Interactive Visualization:** Create interactive plots using libraries like Plotly or Bokeh.
  * **More Robust Optimization:** Explore different optimization algorithms and consider using a more robust initialization strategy.
  * **Cross-Validation:** Implement cross-validation to assess model generalization performance.

**License:**

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/url?sa=E&source=gmail&q=LICENSE) file for details.  (You'll need to create a LICENSE file with the MIT License text.)
