from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as stattools
from alpha_vantage.timeseries import TimeSeries
from fracdiff import fdiff
from fracdiff.sklearn import FracdiffStat
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Pull intraday stock data
with open("key.txt") as f:
    key = f.readline().strip("\n")

sym = 'AAPL'
tseries = TimeSeries(key=key, output_format='pandas')
data, meta_data = tseries.get_intraday(symbol=sym,
                                       interval='60min',
                                       outputsize='full')

close = pd.Series(np.log(data['4. close'][::-1]))

plt.plot(np.array(close))
plt.show()


def adfstat(d):
    diff = fdiff(close, d, mode="valid")
    stat, *_ = stattools.adfuller(diff)
    return stat


# noinspection PyShadowingNames
def correlation(d):
    diff = fdiff(close, d, mode="valid")
    corr = np.corrcoef(close[-diff.size:], diff)[0, 1]
    return corr


ds = np.linspace(0.0, 1.0, 10)
stats = np.vectorize(adfstat)(ds)
corrs = np.vectorize(correlation)(ds)

# 5% critical value of stationarity
_, _, _, _, crit, _ = stattools.adfuller(close)

# plot
fig, ax_stat = plt.subplots(figsize=(24, 8))
ax_corr = ax_stat.twinx()

ax_stat.plot(ds, stats, color="blue", label="ADF statistics (left)")
ax_corr.plot(ds, corrs, color="orange", label="correlation (right)")
ax_stat.axhline(y=crit["5%"],
                linestyle="--",
                color="k",
                label="5% critical value")

plt.title("Stationarity and memory of fractionally differentiated " + sym)
fig.legend()
plt.show()

# Find exact min diff order at 5% level
X = close.values.reshape(-1, 1)
fs = FracdiffStat(mode="valid")

Xdiff = fs.fit_transform(X)
_, pvalue, _, _, _, _ = stattools.adfuller(Xdiff.reshape(-1))
corr = np.corrcoef(X[-Xdiff.size:, 0], Xdiff.reshape(-1))[0][1]

print("* Order: {:.2f}".format(fs.d_[0]))
print("* ADF p-value: {:.2f} %".format(100 * pvalue))
print("* Correlation with the original time-series: {:.2f}".format(corr))

# Plot original seres against min fracdiff order for stationarity
close_diff = pd.Series(Xdiff.reshape(-1), index=close.index[-Xdiff.size:])

fig, ax_s = plt.subplots(figsize=(24, 8))
plt.title("S&P 500 and its Memory Preserving Differentiation")
ax_d = ax_s.twinx()

plot_s = ax_s.plot(close, color="blue", linewidth=0.4, label=sym)
plot_d = ax_d.plot(
    close_diff,
    color="orange",
    linewidth=0.4,
    label=sym + f", {fs.d_[0]:.2f} order difference",
)
plots = plot_s + plot_d

ax_s.legend(plots, [p.get_label() for p in plots], loc=0)
plt.show()

df = close_diff

n_steps = 5
features = []
labels = []
for i in range(len(df) - n_steps):
    features.append(df.iloc[i:i + n_steps].values)
    labels.append(df.iloc[i + n_steps])

features = np.array(features)
labels = np.array(labels)

X = (features - features.mean()) / features.std()
y = (labels - labels.mean()) / labels.std()

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train the model
e_net = ElasticNet(alpha=0.15)
e_net.fit(X_train, y_train)

# calculate the prediction and mean square error
y_pred_elastic = e_net.predict(X_test)
mean_squared_error = np.mean((y_pred_elastic - y_test) ** 2)
print("Mean Squared Error on test set", mean_squared_error)

plt.plot(y_pred_elastic)
plt.plot(y_test)
plt.show()


def sarimax_(signal, fd=False):
    best_arima = None
    src = signal

    if fd is False:
        df = src.diff()
        df_ar = pd.DataFrame(df)
    if fd is True:
        df_ar = pd.DataFrame(src)

    def reverse_close(array):
        if fd is False:
            return array + src.shift(1)
        if fd is True:
            return array

    Qs = range(0, 2)
    qs = range(0, 3)
    Ps = range(0, 3)
    ps = range(0, 3)
    D = 1
    parameters = product(ps, qs, Ps, Qs)
    parameters_list = list(parameters)
    best_aic = float("inf")
    for first, second, third, fourth in parameters_list:
        try:
            arima = SARIMAX(df_ar.values,
                            order=(first, D, second),
                            seasonal_order=(third,
                                            D,
                                            fourth, 4)).fit(disp=True,
                                                            maxiter=200,
                                                            method='powell')
        except ValueError:
            continue
        aic = arima.aic
        if aic < best_aic and aic:
            best_arima = arima
            best_aic = aic

    return reverse_close(best_arima.predict())


sarima = sarimax_(close, fd=False)
sarima_fd = sarimax_(close_diff, fd=True)


mean_squared_error_d = np.mean((sarima[29:] - close[29:]) ** 2)
print("Mean Squared Error on test set", mean_squared_error_d)
plt.plot(np.array(sarima))
plt.plot(np.array(close))
plt.show()


mean_squared_error_fd = np.mean((sarima_fd[20:] - close_diff[20:]) ** 2)
print("Mean Squared Error on test set", mean_squared_error_fd)
plt.plot(np.array(sarima_fd[20:]))
plt.plot(np.array(close_diff[20:]))
plt.show()
