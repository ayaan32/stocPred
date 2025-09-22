import pyreason as pr
import networkx as nx
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import json

# Download historical stock prices
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
data = yf.download(tickers, start="2025-08-01", end="2025-09-15")

prices = data['Adj Close'] if 'Adj Close' in data else data['Close']
prices = prices.dropna()
stocks = prices.columns.tolist()
print("\n--- Historical Prices ---")
print(prices)

# Compute ML-based correlations (R²)
g = nx.DiGraph()
g.add_nodes_from(stocks)
correlations = {}

for i, stock1 in enumerate(stocks):
    for j, stock2 in enumerate(stocks):
        if i != j:
            X = prices[[stock2]].values
            y = prices[stock1].values
            model = LinearRegression()
            model.fit(X, y)
            r2 = model.score(X, y)
            correlations[(stock1, stock2)] = r2
            if r2 > 0.5:  # threshold for meaningful correlation
                g.add_edge(stock1, stock2, correlation=round(r2, 2))

pr.load_graph(g)
print("\n--- ML Learned Correlations (R² values) ---")
for (s1, s2), corr in correlations.items():
    print(f"{s1} <- {s2}: R² = {corr:.2f}")

# Add PyReason rules
pr.reset()
pr.load_graph(g)
pr.add_rule(pr.Rule('uptrend(x) <-0.9 uptrend(y), correlation(x,y)', 'uptrend_rule'))
pr.add_rule(pr.Rule('downtrend(x) <-0.9 downtrend(y), correlation(x,y)', 'downtrend_rule'))

# Initial trends 
initial_trends = {
    'AAPL': 'uptrend',
    'MSFT': 'downtrend',
    'GOOGL': 'downtrend',
    'AMZN': 'uptrend'
}
for stock, trend in initial_trends.items():
    if trend == 'uptrend':
        pr.add_fact(pr.Fact(f'uptrend({stock})', f'uptrend_{stock.lower()}', 0, 2))
    elif trend == 'downtrend':
        pr.add_fact(pr.Fact(f'downtrend({stock})', f'downtrend_{stock.lower()}', 0, 2))

# Run reasoning
interpretation = pr.reason(timesteps=3, convergence_threshold=0.0)
dataframes = pr.filter_and_sort_nodes(interpretation, ['uptrend', 'downtrend'])

# Price update function
def update_prices(prices, trends_df, timestep, graph):
    new_prices = prices.copy()
    for _, row in trends_df.iterrows():
        stock = row['component']
        uptrend = sum(row['uptrend']) / 2 if isinstance(row['uptrend'], list) else 0
        downtrend = sum(row['downtrend']) / 2 if isinstance(row['downtrend'], list) else 0
        net_trend = uptrend - downtrend
        max_corr = max([graph[pred][stock].get('correlation', 0) 
                        for pred in graph.predecessors(stock)], default=1.0)
        new_prices[stock] *= (1 + 0.03 * net_trend * max_corr)
    return new_prices

# Predict prices over timesteps
current_prices = prices.iloc[-1].to_dict() 
predicted_dates = ['Sept 16, 2025', 'Sept 17, 2025', 'Sept 18, 2025']

for t, df in enumerate(dataframes[1:]):  # skip t=0
    print(f'\n--- TIMESTEP {t+1} (Predicted for {predicted_dates[t]}) ---')
    print(df)
    current_prices = update_prices(current_prices, df, t+1, g)
    print("Predicted Prices:")
    for stock, price in current_prices.items():
        print(f"{stock}: ${price:.2f}")

# Save results
results = [{'timestep': t, 'data': df.to_dict()} for t, df in enumerate(dataframes)]
with open('stock_results.json', 'w') as f:
    json.dump(results, f, indent=2)

pr.save_rule_trace(interpretation)

