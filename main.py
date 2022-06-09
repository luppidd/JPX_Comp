import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import logging

logging.basicConfig(level=logging.DEBUG)
stock_prices = pd.read_csv('jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv', index_col =False)

print(f"original stock prices: {len(stock_prices)}")
cleaned_stock_prices = stock_prices[stock_prices['Close'].notna()]

targets = pd.pivot_table(stock_prices, values="Target", index="Date", columns="securitiesCode")
closes = pd.pivot_table(stock_prices, values="Close", index="Date", columns="SecuritiesCode").ffill()
highs = pd.pivot_table(stock_prices, values="Open", index="Date", columns="SecuritiesCode").ffill()
lows = pd.pivot_table(stock_prices, values="Open", index="Date", columns="SecuritiesCode").ffill()
volumes = pd.pivot_table(stock_prices, values="Open", index="Date", columns="SecuritiesCode").ffill()

def ror(window):