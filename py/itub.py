import yfinance as yf

ticker = "ITUB4.SA"
start = "2020-01-01"
end = "2025-01-01"

# Com ajuste (reinveste dividendos)
data_adjusted = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
ret_adjusted = (data_adjusted['Close'][-1] / data_adjusted['Close'][0] - 1) * 100

# Sem ajuste (apenas variação de preço)
data_raw = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
ret_raw = (data_raw['Close'][-1] / data_raw['Close'][0] - 1) * 100

print(f"Retorno COM reinvestimento: {ret_adjusted:.2f}%")
print(f"Retorno SEM reinvestimento: {ret_raw:.2f}%")
print(f"Diferença (dividendos): {ret_adjusted - ret_raw:.2f}%")