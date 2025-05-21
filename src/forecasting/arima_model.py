import logging

import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ARIMAModel:
    def __init__(self, p=2, d=1, q=2, P=1, D=1, Q=1, s=24, auto=False):
        """
        Parameters:
        - p, d, q: Порядки авторегрессии, дифференцирования и скользящего среднего.
        - P, D, Q: Сезонные порядки.
        - s: Сезонный период (по умолчанию 24 часа).
        - auto: Если True, использовать auto_arima для подбора параметров
        """
        self.p, self.d, self.q = p, d, q
        self.P, self.D, self.Q = P, D, Q
        self.s = s
        self.auto = auto
        self.model = None
        self.fitted = None
        self.last_date = None
        self.offset = 0

    def _check_stationarity(self, series):
        # Проверка стационарности с помощью теста Дики-Фуллера
        result = adfuller(series.dropna())
        logging.info(f'ADF Statistic: {result[0]:.4f}, p-value: {result[1]:.4f}')
        return result[1] < 0.05

    def _plot_acf_pacf(self, series):
        #графики ACF и PACF
        diff = series.diff().dropna() if self.d > 0 else series
        if self.D > 0:
            diff = diff.diff(self.s).dropna()

        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plot_acf(diff, ax=plt.gca(), lags=min(len(diff)//2, self.s*2))
        plt.title('ACF (дифференцированный ряд)')
        plt.subplot(122)
        plot_pacf(diff, ax=plt.gca(), lags=min(len(diff)//2, self.s*2))
        plt.title('PACF (дифференцированный ряд)')
        plt.savefig('acf_pacf.png')
        plt.close()

    def _fit_model(self, series):
        # Обучение модели
        self.offset = -series.min() if series.min() < 0 else 0
        transformed_series = series + self.offset
        if self.auto:
            self.model = auto_arima(transformed_series, seasonal=True, m=self.s,
                                    max_p=3, max_d=2, max_q=3,
                                    max_P=2, max_D=1, max_Q=2,
                                    trace=True, error_action='ignore',
                                    suppress_warnings=True)
            self.p, self.d, self.q = self.model.order
            self.P, self.D, self.Q, self.s = self.model.seasonal_order
        else:
            self.model = ARIMA(transformed_series, order=(self.p, self.d, self.q),
                               seasonal_order=(self.P, self.D, self.Q, self.s))
        self.fitted = self.model.fit()
        logging.info(f'Параметры модели: ARIMA({self.p},{self.d},{self.q})x({self.P},{self.D},{self.Q},{self.s})')

    def train(self, data, target='A_plus', group='uuid'):
        if isinstance(data, pd.Series):
            series = data.dropna()
        else:
            series = data[target].dropna()

        if len(series) < self.s * 2:
            logging.warning(f'Недостаточно данных {len(series)} наблюдений')
            return

        if isinstance(series.index, pd.DatetimeIndex):
            series.index.freq = 'h'

        if not self._check_stationarity(series):
            logging.warning('Ряд нестационарен, результаты могут быть ненадежными')

        self._plot_acf_pacf(series)
        self._fit_model(series)
        self.last_date = series.index[-1] if isinstance(series.index, pd.DatetimeIndex) else None
        self.check_residuals()

    def forecast(self, steps):
        #Прогнозирование
        if self.fitted is None:
            raise ValueError('Модель не обучена')
        forecast = self.fitted.forecast(steps=steps)
        return forecast - self.offset

    def check_residuals(self):
        if self.fitted is None:
            return
        residuals = self.fitted.resid
        lb_test = acorr_ljungbox(residuals, lags=[1, 12, self.s], return_df=True)
        logging.info(f"Тест Льюнга-Бокса: {lb_test}")
        if lb_test['lb_pvalue'].min() < 0.05:
            logging.warning("Обнаружена автокорреляция в остатках, модель может быть неадекватной")
        plt.figure(figsize=(12, 6))
        plt.subplot(211)
        residuals.plot(title='Остатки')
        plt.subplot(212)
        plot_acf(residuals, lags=self.s, ax=plt.gca())
        plt.title('ACF остатков')
        plt.savefig('residuals.png')
        plt.close()

