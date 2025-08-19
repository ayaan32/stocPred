import pyreason as pr
import networkx as nx
import pandas as pd

# Reset PyReason environment
pr.reset()

# Create a stock correlation graph
g = nx.DiGraph()
stocks = ['Apple', 'Microsoft', 'Google', 'Amazon']
g.add_nodes_from(stocks)

# Add edges with correlation weights
g.add_edge('Apple', 'Microsoft', correlation=0.8)
g.add_edge('Microsoft', 'Google', correlation=0.9)
g.add_edge('Google', 'Amazon', correlation=0.7)
g.add_edge('Apple', 'Amazon', correlation=0.6)

# Load graph into PyReason
pr.load_graph(g)

# Define rules for price trend propagation
# pr.add_rule(pr.Rule('uptrend(x) <-1 uptrend(y), correlation(x,y)', 'uptrend_rule'))
# pr.add_rule(pr.Rule('downtrend(x) <-1 downtrend(y), correlation(x,y)', 'downtrend_rule'))

pr.add_rule(pr.Rule('uptrend(x) <- correlation(x,y) uptrend(y)', 'uptrend_rule'))
pr.add_rule(pr.Rule('downtrend(x) <- correlation(x,y) downtrend(y)', 'downtrend_rule'))

# Add initial facts
pr.add_fact(pr.Fact('uptrend(Microsoft)', 'uptrend_msft', 0, 2))
pr.add_fact(pr.Fact('downtrend(Amazon)', 'downtrend_amzn', 0, 2))

# Run reasoning for 3 timesteps
interpretation = pr.reason(timesteps=3)

# Initialize hypothetical stock prices
stock_prices = {
    'Apple': 150.0,
    'Microsoft': 300.0,
    'Google': 2500.0,
    'Amazon': 3200.0
}

# Function to update prices based on trends
def update_prices(prices, trends_df, timestep):
    new_prices = prices.copy()
    for _, row in trends_df.iterrows():
        stock = row['component']
        uptrend = row['uptrend']
        downtrend = row['downtrend']
        # If uptrend is definitive ([1.0, 1.0]), increase price by 5%
        if uptrend == [1.0, 1.0]:
            new_prices[stock] *= 1.05
        # If downtrend is definitive ([1.0, 1.0]), decrease price by 5%
        if downtrend == [1.0, 1.0]:
            new_prices[stock] *= 0.95
    return new_prices

# Filter and sort nodes, ensuring all stocks are included
dataframes = pr.filter_and_sort_nodes(interpretation, ['uptrend', 'downtrend'])

# Process and print results with prices
current_prices = stock_prices.copy()
for t, df in enumerate(dataframes):
    print(f'\nTIMESTEP - {t}')
    print(df)
    # Update prices based on trends
    current_prices = update_prices(current_prices, df, t)
    # Print stock prices
    print("Stock Prices:")
    for stock, price in current_prices.items():
        print(f"{stock}: ${price:.2f}")

# Save rule trace for debugging
pr.save_rule_trace(interpretation)
 