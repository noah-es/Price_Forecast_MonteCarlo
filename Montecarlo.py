{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "825c4b0a-082b-46eb-a997-4cbcaff05963",
   "metadata": {},
   "source": [
    "## Teoría\n",
    "https://gsnchez.com/blog/article/El-metodo-de-monte-carlo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "686c49cd-c60b-4018-a2fd-52d0b75b3ef9",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import yfinance as yf\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "from matplotlib import style\n",
    "from scipy.stats import norm\n",
    "\n",
    "style.use('seaborn-v0_8')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "a9e5b689-9a56-4824-b593-0d60bfd19d6f",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[*********************100%%**********************]  1 of 1 completed\n",
      "\n",
      "1 Failed download:\n",
      "['META']: ConnectionError(MaxRetryError('HTTPSConnectionPool(host=\\'query2.finance.yahoo.com\\', port=443): Max retries exceeded with url: /v8/finance/chart/%ticker%?period1=1325394000&period2=1722276005&interval=1d&includePrePost=False&events=div%2Csplits%2CcapitalGains&crumb=eQlxBPVCvNc (Caused by NameResolutionError(\"<urllib3.connection.HTTPSConnection object at 0x000001DA65B3B090>: Failed to resolve \\'query2.finance.yahoo.com\\' ([Errno 11001] getaddrinfo failed)\"))'))\n"
     ]
    }
   ],
   "source": [
    "ticker = 'META'\n",
    "data = pd.DataFrame()\n",
    "data[ticker] = yf.download(ticker, start='2012-1-1')['Adj Close']\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "a81d0c70-4f08-410d-8dd3-ccfb4431059d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>META</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "Empty DataFrame\n",
       "Columns: [META]\n",
       "Index: []"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "c1d6da85-6b23-415a-af67-7c66cab41b44",
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "attempt to get argmax of an empty sequence",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[37], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m log_returns \u001b[38;5;241m=\u001b[39m np\u001b[38;5;241m.\u001b[39mlog(\u001b[38;5;241m1\u001b[39m \u001b[38;5;241m+\u001b[39m data\u001b[38;5;241m.\u001b[39mpct_change())\n\u001b[0;32m      2\u001b[0m media \u001b[38;5;241m=\u001b[39m log_returns\u001b[38;5;241m.\u001b[39mmean()\n\u001b[0;32m      3\u001b[0m var \u001b[38;5;241m=\u001b[39m log_returns\u001b[38;5;241m.\u001b[39mvar()\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\pandas\\core\\generic.py:11712\u001b[0m, in \u001b[0;36mNDFrame.pct_change\u001b[1;34m(self, periods, fill_method, limit, freq, **kwargs)\u001b[0m\n\u001b[0;32m  11710\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m _, col \u001b[38;5;129;01min\u001b[39;00m cols:\n\u001b[0;32m  11711\u001b[0m     mask \u001b[38;5;241m=\u001b[39m col\u001b[38;5;241m.\u001b[39misna()\u001b[38;5;241m.\u001b[39mvalues\n\u001b[1;32m> 11712\u001b[0m     mask \u001b[38;5;241m=\u001b[39m mask[np\u001b[38;5;241m.\u001b[39margmax(\u001b[38;5;241m~\u001b[39mmask) :]\n\u001b[0;32m  11713\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m mask\u001b[38;5;241m.\u001b[39many():\n\u001b[0;32m  11714\u001b[0m         warnings\u001b[38;5;241m.\u001b[39mwarn(\n\u001b[0;32m  11715\u001b[0m             \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mThe default fill_method=\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mpad\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m in \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m  11716\u001b[0m             \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;132;01m{\u001b[39;00m\u001b[38;5;28mtype\u001b[39m(\u001b[38;5;28mself\u001b[39m)\u001b[38;5;241m.\u001b[39m\u001b[38;5;18m__name__\u001b[39m\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m.pct_change is deprecated and will \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m  11721\u001b[0m             stacklevel\u001b[38;5;241m=\u001b[39mfind_stack_level(),\n\u001b[0;32m  11722\u001b[0m         )\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\numpy\\core\\fromnumeric.py:1229\u001b[0m, in \u001b[0;36margmax\u001b[1;34m(a, axis, out, keepdims)\u001b[0m\n\u001b[0;32m   1142\u001b[0m \u001b[38;5;250m\u001b[39m\u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m   1143\u001b[0m \u001b[38;5;124;03mReturns the indices of the maximum values along an axis.\u001b[39;00m\n\u001b[0;32m   1144\u001b[0m \n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m   1226\u001b[0m \u001b[38;5;124;03m(2, 1, 4)\u001b[39;00m\n\u001b[0;32m   1227\u001b[0m \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m   1228\u001b[0m kwds \u001b[38;5;241m=\u001b[39m {\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mkeepdims\u001b[39m\u001b[38;5;124m'\u001b[39m: keepdims} \u001b[38;5;28;01mif\u001b[39;00m keepdims \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m np\u001b[38;5;241m.\u001b[39m_NoValue \u001b[38;5;28;01melse\u001b[39;00m {}\n\u001b[1;32m-> 1229\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m _wrapfunc(a, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124margmax\u001b[39m\u001b[38;5;124m'\u001b[39m, axis\u001b[38;5;241m=\u001b[39maxis, out\u001b[38;5;241m=\u001b[39mout, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwds)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\numpy\\core\\fromnumeric.py:59\u001b[0m, in \u001b[0;36m_wrapfunc\u001b[1;34m(obj, method, *args, **kwds)\u001b[0m\n\u001b[0;32m     56\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m _wrapit(obj, method, \u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwds)\n\u001b[0;32m     58\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[1;32m---> 59\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m bound(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwds)\n\u001b[0;32m     60\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mTypeError\u001b[39;00m:\n\u001b[0;32m     61\u001b[0m     \u001b[38;5;66;03m# A TypeError occurs if the object does have such a method in its\u001b[39;00m\n\u001b[0;32m     62\u001b[0m     \u001b[38;5;66;03m# class, but its signature is not identical to that of NumPy's. This\u001b[39;00m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m     66\u001b[0m     \u001b[38;5;66;03m# Call _wrapit from within the except clause to ensure a potential\u001b[39;00m\n\u001b[0;32m     67\u001b[0m     \u001b[38;5;66;03m# exception has a traceback chain.\u001b[39;00m\n\u001b[0;32m     68\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m _wrapit(obj, method, \u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwds)\n",
      "\u001b[1;31mValueError\u001b[0m: attempt to get argmax of an empty sequence"
     ]
    }
   ],
   "source": [
    "log_returns = np.log(1 + data.pct_change())\n",
    "media = log_returns.mean()\n",
    "var = log_returns.var()\n",
    "drift = media - (0.5*var)\n",
    "std = log_returns.std()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "daab2b81-a142-48ec-aa94-faa004b6da6a",
   "metadata": {},
   "outputs": [],
   "source": [
    "days = 100\n",
    "n_pruebas = 1000\n",
    "\n",
    "z = norm.ppf(np.random.rand(days,n_pruebas))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a3a3d738-c852-475a-92f2-5a7c921000cf",
   "metadata": {},
   "outputs": [],
   "source": [
    "retornos_diarios = np.exp(drift.values + std.values * z)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a27de9e0-64d5-43eb-9706-049554600268",
   "metadata": {},
   "outputs": [],
   "source": [
    "camino_de_precios = np.zeros_like(retornos_diarios)\n",
    "camino_de_precios[0] = data.iloc[-1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ae46400c-4a7f-424e-abed-508d6d39dc3d",
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in range(1,days):\n",
    "    camino_de_precios[i] = camino_de_precios[i-1]*retornos_diarios[i]\n",
    "\n",
    "plt.figure(figsize = (15,6))\n",
    "plt.plot(pd.DataFrame(camino_de_precios))\n",
    "plt.xlabel('Numero de Días')\n",
    "plt.ylabel('Precio de ' + ticker)\n",
    "plt.show()\n",
    "sns.histplot(pd.DataFrame(camino_de_precios).iloc[-1])\n",
    "plt.xlabel('Precio a ' + str(days) + ' días')\n",
    "plt.ylabel('Frecuencia')\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
