import yfinance as yf
import numpy as np

def get_binary_returns(ticker="SPY", period="2y"):
    """
    Download stock data and convert daily returns
    to binary signals:
      1 = price went UP today
      0 = price went DOWN or flat today
    """
    df = yf.download(ticker, period=period, auto_adjust=True)

    # Daily return: positive or negative
    df["return"] = df["Close"].pct_change()
    df.dropna(inplace=True)

    # Binary signal: 1 if up, 0 if down/flat
    binary = (df["return"] > 0).astype(int).values.tolist()

    return binary

if __name__ == "__main__":
    signals = get_binary_returns()
    print(f"Total trading days: {len(signals)}")
    print(f"Up days:   {sum(signals)}")
    print(f"Down days: {len(signals) - sum(signals)}")
    print(f"First 20 signals: {signals[:20]}")

    # Save for later use
    np.save("data/spy_signals.npy", signals)
    print("Saved to data/spy_signals.npy")