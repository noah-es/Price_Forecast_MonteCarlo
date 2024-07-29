import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.stats import norm

style.use('seaborn-v0_8')

ticker = 'META'
data = pd.DataFrame()
data[ticker] = yf.download(ticker, start='2012-1-1')['Adj Close']

log_returns = np.log(1 + data.pct_change())
media = log_returns.mean()
var = log_returns.var()
drift = media - (0.5*var)
std = log_returns.std()

days = 100
n_pruebas = 1000

z = norm.ppf(np.random.rand(days,n_pruebas))
retornos_diarios = np.exp(drift.values + std.values * z)
camino_de_precios = np.zeros_like(retornos_diarios)
camino_de_precios[0] = data.iloc[-1]

for i in range(1,days):
    camino_de_precios[i] = camino_de_precios[i-1]*retornos_diarios[i]

plt.figure(figsize = (15,6))
plt.plot(pd.DataFrame(camino_de_precios))
plt.xlabel('Numero de Días')
plt.ylabel('Precio de ' + ticker)
plt.show()
sns.histplot(pd.DataFrame(camino_de_precios).iloc[-1])
plt.xlabel('Precio a ' + str(days) + ' días')
plt.ylabel('Frecuencia')
plt.show()
