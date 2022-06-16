import pandas as pd
import numpy as np
from tqdm import tqdm
from lightgbm import LGBMRegressor
import plotly.graph_objects as go

def calc_return(X,targets, keep_n):
    '''My home made function to do the weighted score'''
    n_stocks = X.shape[1]
    longs = (X<keep_n)*1
    shorts = (X>(n_stocks- keep_n - 1))*1

    if keep_n>1:
        longs = ((1-(X/(keep_n-1)))+1)*longs/(3*keep_n)
        shorts = -((X-(n_stocks- keep_n))/(keep_n-1)+1)*shorts/(3*keep_n)
    else:
        longs = longs/2
        shorts = -shorts/2
    return (targets*(shorts+longs)).sum(axis=1)

df = pd.read_csv(r'D:\PycharmProjects\Kaggle\JPX_Comp\temp\train_files\secondary_stock_prices_may262022.csv')
df.Date = pd.to_datetime(df.Date)

targets = pd.pivot_table(df, values = "Target", index = "Date", columns = "SecuritiesCode")
closes = pd.pivot_table(df, values = "Close", index = "Date", columns = "SecuritiesCode").ffill()
opens = pd.pivot_table(df, values = "Open", index = "Date", columns = "SecuritiesCode").ffill()
highs = pd.pivot_table(df, values = "High", index = "Date", columns = "SecuritiesCode").ffill()
lows = pd.pivot_table(df, values = "Low", index = "Date", columns = "SecuritiesCode").ffill()
volumes = pd.pivot_table(df, values = "Volume", index = "Date", columns = "SecuritiesCode").ffill()


def ror(window):
    return pd.melt((closes - closes.shift(window)) / closes.shift(window),
                   ignore_index=False).reset_index().dropna().rename(columns={"value": f"ror_{window}"})


def vol(window):
    return pd.melt(volumes.rolling(window).mean(), ignore_index=False).reset_index().dropna().rename(
        columns={"value": f"vol_{window}"})


def atr(window):
    a = highs - lows
    b = abs(highs - closes.shift(1))
    c = abs(lows - closes.shift(1))
    return pd.melt(pd.DataFrame(np.max([a, b, c], axis=0), index=a.index, columns=a.columns).rolling(window).mean(),
                   ignore_index=False).reset_index().dropna().rename(columns={"value": f"atr_{window}"})


def atr_day(window):
    a = highs - lows

    return pd.melt(a.rolling(window).mean(), ignore_index=False).reset_index().dropna().rename(
        columns={"value": f"atrday_{window}"})


def atr_gap(window):
    a = abs(highs - closes.shift(1))

    return pd.melt(a.rolling(window).mean(), ignore_index=False).reset_index().dropna().rename(
        columns={"value": f"atrgap_{window}"})


def atr_hige(window):
    a = abs(lows - closes.shift(1))

    return pd.melt(a.rolling(window).mean(), ignore_index=False).reset_index().dropna().rename(
        columns={"value": f"atrhige_{window}"})


def dev(window):
    return pd.melt((closes.diff() / closes.shift(1)).rolling(window).std(),
                   ignore_index=False).reset_index().dropna().rename(columns={"value": f"variation_{window}"})


def HL(window):
    return pd.melt((highs.rolling(window).max() - lows.rolling(window).min()),
                   ignore_index=False).reset_index().dropna().rename(columns={"value": f"HL_{window}"})


def market_impact(window):
    return pd.melt((closes.diff() / volumes).rolling(window).mean(), ignore_index=False).reset_index().dropna().rename(
        columns={"value": f"market_imact_{window}"})


features = df[["Date", "SecuritiesCode", "Close", "ExpectedDividend"]].fillna(0)

for func in [ror, vol, atr, atr_day, atr_gap, atr_hige, HL, market_impact]:
    for window in tqdm([1, 5, 10, 20, 40, 60, 100]):
        features = pd.merge(features, func(window), on=["Date", "SecuritiesCode"], how="left")

for window in tqdm([2, 5, 10, 20, 40, 60, 100]):
    features = pd.merge(features, dev(window), on=["Date", "SecuritiesCode"], how="left")

features["vol_diff"] = features["vol_20"].diff()
features["atr_diff"] = features["atr_20"].diff()

X = features.fillna(0)
y = df["Target"]

Xtrain = X.loc[(X.Date<"2022-01-01")].drop("Date",axis=1).astype(np.float32)
ytrain = y.loc[Xtrain.index]

m = LGBMRegressor()
m.fit(Xtrain,ytrain)

subdf = df.loc[(df.Date>"2022-01-01")]
Xtest = X.loc[(X.Date>"2022-01-01")].drop("Date",axis=1).astype(np.float32)

targets_2022plus = pd.pivot_table(subdf, index = "Date", columns = "SecuritiesCode", values="Target")
subdf["predictions"] = m.predict(Xtest)
preds = pd.pivot_table(subdf, index = "Date", columns = "SecuritiesCode", values="predictions").reindex(targets_2022plus.index).ffill().bfill()

buy_sell_2022plus = 1999-np.argsort(np.argsort(preds))
portfolio_return = calc_return(buy_sell_2022plus, targets_2022plus, 200)
average_index_return = targets_2022plus.mean(axis=1)
print(f"daily sharp ratio, lgb model : {np.round(portfolio_return.mean()/portfolio_return.std(),3)}")
print(f"daily sharp ratio, average index: {np.round(average_index_return.mean()/portfolio_return.std(),3)}")
