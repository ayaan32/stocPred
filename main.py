import pyreason as pr
import networkx as nx
import pandas as pd
import json

pr.reset()

# stock correlation graph
g = nx.DiGraph()
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
g.add_nodes_from(stocks)

# Correlations from historical data (Aug 1 to Aug 13, 2025)
correlations = {
    ('AAPL', 'MSFT'): 0.87,
    ('GOOGL', 'AMZN'): 0.75,
    ('AAPL', 'GOOGL'): 0.56,  
    ('MSFT', 'AMZN'): 0.82
}
for (stock1, stock2), corr in correlations.items():
    if corr > 0.5:
        g.add_edge(stock1, stock2, correlation=corr)

pr.load_graph(g)

# Rules
pr.add_rule(pr.Rule('uptrend(x) <-0.9 uptrend(y), correlation(x,y)', 'uptrend_rule'))
pr.add_rule(pr.Rule('downtrend(x) <-0.9 downtrend(y), correlation(x,y)', 'downtrend_rule'))

# Initial trends based on Aug 12 to Aug 13 change
initial_trends = {
    'AAPL': 'uptrend',
    'MSFT': 'downtrend',
    'GOOGL': 'downtrend',
    'AMZN': 'uptrend'
}

# Add initial facts for t=0 to 2
for stock, trend in initial_trends.items():
    if trend == 'uptrend':
        pr.add_fact(pr.Fact(f'uptrend({stock})', f'uptrend_{stock.lower()}', 0, 2))
    elif trend == 'downtrend':
        pr.add_fact(pr.Fact(f'downtrend({stock})', f'downtrend_{stock.lower()}', 0, 2))

# Run reasoning for 3 timesteps (predicting Aug 14, 15, 18)
interpretation = pr.reason(timesteps=3)

# Initial prices from Aug 13, 2025
stock_prices = {
    'AAPL': 233.33,
    'MSFT': 520.58,
    'GOOGL': 201.96,
    'AMZN': 224.56
}

# Actual prices for comparison
actual_prices = {
    'Aug 14, 2025': {'AAPL': 232.78, 'MSFT': 522.48, 'GOOGL': 202.94, 'AMZN': 230.98},
    'Aug 15, 2025': {'AAPL': 231.59, 'MSFT': 520.17, 'GOOGL': 203.90, 'AMZN': 231.03},
    'Aug 18, 2025': {'AAPL': 230.89, 'MSFT': 517.10, 'GOOGL': 203.50, 'AMZN': 231.49}
}

# update prices based on trends
def update_prices(prices, trends_df, timestep, graph):
    new_prices = prices.copy()
    for _, row in trends_df.iterrows():
        stock = row['component']
        uptrend = sum(row['uptrend']) / 2 if isinstance(row['uptrend'], list) else 0
        downtrend = sum(row['downtrend']) / 2 if isinstance(row['downtrend'], list) else 0
        net_trend = uptrend - downtrend
        max_corr = max([graph[pred][stock].get('correlation', 0) for pred in graph.predecessors(stock)], default=1.0)
        new_prices[stock] *= (1 + 0.03 * net_trend * max_corr)
    return new_prices

# Filter and sort nodes
dataframes = pr.filter_and_sort_nodes(interpretation, ['uptrend', 'downtrend'])

# Save results to JSON
results = [{'timestep': t, 'data': df.to_dict()} for t, df in enumerate(dataframes)]
with open('/Users/ayaantariq/Documents/stock_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Process and print results with prices
current_prices = stock_prices.copy()
predicted_dates = ['Aug 14, 2025', 'Aug 15, 2025', 'Aug 18, 2025']
for t, df in enumerate(dataframes[1:]):
    print(f'\nTIMESTEP - {t+1} (Predicted for {predicted_dates[t]})')
    print(df)
    current_prices = update_prices(current_prices, df, t+1, g)
    print("Predicted Stock Prices:")
    for stock, price in current_prices.items():
        print(f"{stock}: ${price:.2f}")
    print("Actual Stock Prices:")
    actual = actual_prices[predicted_dates[t]]
    for stock, price in actual.items():
        print(f"{stock}: ${price:.2f}")

# Save rule trace for debugging
pr.save_rule_trace(interpretation)