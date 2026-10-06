import yfinance as yf

data = yf.download(
    "AAPL",
    period="1mo",
    auto_adjust=True,
    progress=False
)

print(data.head())