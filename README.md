PyReason Stock Trend Prediction Demo
A demo using PyReason to predict stock price trends for AAPL, MSFT, GOOGL, and AMZN based on historical correlations and price data from August 1–13, 2025. It forecasts trends for August 14–18, 2025, and compares predicted prices to actual values, leveraging PyReason’s graph-based and temporal reasoning capabilities.
Setup

Install Python 3.10 and dependencies:/opt/homebrew/bin/python3.10 -m pip install pyreason pandas


Run the PyReason script:/opt/homebrew/bin/python3.10 stock_trend_prediction.py



Output

Generates dataframes showing uptrend/downtrend scores for each stock across three timesteps.
Produces stock_results.json with trend data and predicted prices.
Saves a rule trace (rule_trace_nodes_*.csv) for debugging and interpretability.

Relevance
Demonstrates PyReason’s rule-based reasoning for financial trend analysis, inspired by Dr. Paulo Shakarian’s work in neurosymbolic AI and graph-based diffusion. Uses real historical data to ground predictions, with a focus on interpretability via rule traces.
Limitations

Simplified price update model (3% change scaled by trend strength and correlation).
Sparse correlation graph limits trend propagation.
Predictions are directional and may not match actual price magnitudes.
