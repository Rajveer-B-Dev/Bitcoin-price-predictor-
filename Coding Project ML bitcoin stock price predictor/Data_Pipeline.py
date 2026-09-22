import pandas as pd
import numpy as np

def add_Indicators(Data_Files, n):
    Data = Data_Files

    N = 14

    Price_Change = Data["Close"].diff()
    closing_price = Data["Close"]
    high_price = Data["High"]
    low_price = Data["Low"]

    Gain = Price_Change.clip(lower=0)
    Loss = -Price_Change.clip(upper=0)

    avg_gain = Gain.rolling(window=N).mean()
    avg_loss = Loss.rolling(window=N).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1+rs))

    SCP = closing_price.rolling(window=N).sum()
    sma = (1/N)*SCP

    a = 2/(N+1)
    n_prices = closing_price[0:N]
    Past_price = closing_price.shift(N)
    Current_price = closing_price
    roc = ((Current_price - Past_price)/(Past_price))*100

    TR_data = pd.DataFrame({
        'HL': high_price-low_price,
        'HC': abs(high_price-closing_price.shift(1)),
        'LC': abs(low_price-closing_price.shift(1))})
    TR = TR_data.max(axis=1)
    atr = TR.rolling(window=N).mean()

    initial_ema = n_prices.mean()
    ema = np.full(len(closing_price), np.nan)
    ema[N-1] = initial_ema
    for i in range(N, len(closing_price)):
        ema[i] = (a*Current_price[i]) + ((1-a) * ema[i-1])

        

    Data["RSI"] = rsi
    Data["SMA"] = sma
    Data["ROC"] = roc
    Data["ATR"] = atr
    Data["EMA"] = ema

    SMAt = Data["SMA"]
    EMAt = Data["EMA"]
    crossover = np.full(len(closing_price), np.nan)
    for i in range(N, len(closing_price)):
        crossover[i] = np.where(EMAt[i] > SMAt[i], 1, 0)

    Data["Crossover"] = crossover
    #Data = Data.drop(columns=["ticker"])
    Data = Data.dropna()
    Data = Data.drop(columns=["Close"])
    



    mean_train_X = np.load(f"mean_train_X {n}.npy")
    std_train_X = np.load(f"std_train_X {n}.npy")

    Last_row = Data.iloc[-1]
    Last_row = np.array(Last_row, dtype=float)
    Last_row = Last_row.reshape(1, -1)


    price = ((Last_row - mean_train_X) / std_train_X)
    return price
