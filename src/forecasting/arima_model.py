import logging
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ARIMAModel:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.series = None
        self.series_index = None
        self.order = config['forecasting'].get('arima', {}).get('order', (2, 0, 1))
        self.horizon = config['forecasting'].get('horizon', 24)

    def train(self, train_data):
        try:
            train_data = train_data.set_index('time_dt')
            self.series = train_data['target']
            self.series_index = self.series.index

            self.series = self.series.asfreq('h', method='ffill')
            self.series_index = self.series.index

            if len(self.series) < 2:
                raise ValueError("Недостаточно данных для обучения модели ARIMA")

            logging.info(f"Обучающие данные: {len(self.series)} точек, с {self.series_index[0]} по {self.series_index[-1]}")

            # Обучение модели
            self.model = ARIMA(self.series, order=self.order)
            self.model_fit = self.model.fit()
            logging.info(f"Модель ARIMA обучена с order={self.order}")

        except Exception as e:
            raise ValueError(f"Не удалось обучить модель ARIMA: {str(e)}")

    def forecast(self, horizon):
        try:
            if self.model_fit is None:
                raise ValueError("Модель не обучена.")

            if self.series_index is None or len(self.series_index) == 0:
                raise ValueError("Индекс временного ряда не инициализирован")

            logging.info(f"Создание прогноза ARIMA на горизонт {horizon}")
            # Прогноз
            forecast_result = self.model_fit.forecast(steps=horizon)
            forecast = np.array(forecast_result)

            # Доверительные интервалы
            conf_int = self.model_fit.get_forecast(steps=horizon).conf_int(alpha=0.05)
            conf_int_df = pd.DataFrame({
                'lower': conf_int['lower ' + self.series.name],
                'upper': conf_int['upper ' + self.series.name]
            })

            if len(forecast) != horizon:
                logging.warning(f"Длина прогноза ({len(forecast)}) не соответствует горизонту ({horizon}), корректирую")
                if len(forecast) > horizon:
                    forecast = forecast[:horizon]
                    conf_int_df = conf_int_df.iloc[:horizon]
                else:
                    forecast = np.pad(forecast, (0, horizon - len(forecast)), mode='constant', constant_values=np.nan)
                    conf_int_df = conf_int_df.reindex(range(horizon), fill_value=np.nan)

            logging.info(f"Прогноз ARIMA успешно создан, длина={len(forecast)}")
            return forecast, conf_int_df

        except Exception as e:
            logging.error(f"Ошибка прогнозирования ARIMA: {str(e)}")
            raise ValueError(f"Не удалось выполнить прогноз ARIMA: {str(e)}")

    def evaluate(self, actual, forecast):
        try:
            mae = mean_absolute_error(actual, forecast)
            rmse = np.sqrt(mean_squared_error(actual, forecast))
            logging.info(f"Оценка прогноза: MAE={mae:.4f}, RMSE={rmse:.4f}")
            return mae, rmse
        except Exception as e:
            raise ValueError(f"Не удалось оценить прогноз: {str(e)}")