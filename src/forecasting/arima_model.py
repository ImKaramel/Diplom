import logging
import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from src.preprocessing import TimeSeriesAnalyzer


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*force_all_finite.*ensure_all_finite.*"
)

class ARIMAModel:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.model_fit = None
        self.series = None
        self.series_index = None
        self.order = config['forecasting'].get('arima', {}).get('order', (2, 0, 1))
        self.seasonal_order = config['forecasting'].get('arima', {}).get('seasonal_order', (0, 0, 0, 24))
        self.horizon = config['forecasting'].get('horizon', 24)
        self.analyzer = TimeSeriesAnalyzer(target_column='target', config=config)

    def train(self, train_data):
        try:

            train_data = train_data.set_index('time_dt')
            self.series = train_data['target']
            self.series_index = self.series.index


            if self.series_index.duplicated().any():
                logging.warning(f"Обнаружено {self.series_index.duplicated().sum()} дублирующихся временных меток")
                self.series = self.series[~self.series_index.duplicated(keep='last')]
                self.series_index = self.series.index
                logging.info(f"Дубликаты удалены, осталось {len(self.series)} точек")


            self.series = self.series.asfreq('h').interpolate(method='linear')
            logging.info(f"Частота после asfreq: {pd.infer_freq(self.series.index)}")


            if pd.infer_freq(self.series.index) not in ['H', 'h']:
                logging.warning("Индекс не имеет почасовой частоты, выполняется переиндексация")
                start = self.series.index.min()
                end = self.series.index.max()
                new_index = pd.date_range(start=start, end=end, freq='h')
                self.series = self.series.reindex(new_index).interpolate(method='linear')
                logging.info(f"Частота после переиндексации: {pd.infer_freq(self.series.index)}")

            self.series_index = self.series.index


            min_points = max(24, self.config['preprocessing'].get('lookback', 168) + self.horizon)
            if len(self.series) < min_points:
                raise ValueError(f"Недостаточно данных для обучения: {len(self.series)} точек, требуется минимум {min_points}")

            logging.info(f"Обучающие данные: {len(self.series)} точек, с {self.series_index[0]} до {self.series_index[-1]}")


            diff_series, d = self.analyzer.make_stationary(self.series, max_diff=1, name="Training Series")
            self.series = diff_series
            self.series_index = self.series.index
            logging.info(f"Стационарный ряд получен с дифференцированием порядка {d}")
            logging.info(f"Частота после make_stationary: {pd.infer_freq(self.series.index)}")

            # Адаптивный выбор из трёх комбинаций
            best_aic = float('inf')
            best_order = None
            best_seasonal_order = None

            combinations = [
                ((0, 1, 2), (1, 0, 0, 24)),  # Быстрая и эффективная комбинация
                ((1, 1, 2), (1, 0, 0, 24)),  # Хорошая точность
                ((2, 1, 1), (1, 0, 1, 24))   # Высокая точность при сложных данных
            ]

            for order, seasonal_order in combinations:
                try:
                    model = ARIMA(self.series, order=order, seasonal_order=seasonal_order)
                    model_fit = model.fit()
                    aic = model_fit.aic
                    logging.info(f"Порядок {order}, сезонный порядок {seasonal_order}, AIC={aic:.4f}")
                    if aic < best_aic:
                        best_aic = aic
                        best_order = order
                        best_seasonal_order = seasonal_order
                except Exception as e:
                    logging.warning(f"Ошибка при обучении с {order}, {seasonal_order}: {e}")
                    continue

            if best_order is None:
                raise ValueError("Не удалось найти подходящую комбинацию параметров")

            self.order = best_order
            self.seasonal_order = best_seasonal_order
            self.model = ARIMA(self.series, order=self.order, seasonal_order=self.seasonal_order)
            self.model_fit = self.model.fit()
            logging.info(f"Выбран лучший порядок: order={self.order}, seasonal_order={self.seasonal_order}, AIC={best_aic:.4f}")

        except Exception as e:
            logging.error(f"Ошибка при обучении ARIMA: {e}")
            raise ValueError(f"Не удалось обучить модель ARIMA: {e}")

    def update(self, new_data):
        try:
            if self.model_fit is None:
                raise ValueError("Модель не обучена.")
            if 'target' not in new_data.columns:
                raise ValueError("Столбец 'target' отсутствует в новых данных")
            if 'time_dt' not in new_data.columns:
                raise ValueError("Столбец 'time_dt' отсутствует в новых данных")

            new_data = new_data.set_index('time_dt')
            new_series = new_data['target']


            if new_series.index.duplicated().any():
                logging.warning(f"Обнаружено {new_series.index.duplicated().sum()} дублирующихся временных меток в новых данных")
                new_series = new_series[~new_series.index.duplicated(keep='last')]
                logging.info(f"Дубликаты удалены из новых данных, осталось {len(new_series)} точек")

            self.series = pd.concat([self.series, new_series], axis=0).dropna().asfreq('h').interpolate(method='linear')


            if pd.infer_freq(self.series.index) not in ['H', 'h']:
                logging.warning("Индекс после обновления не имеет почасовой частоты, выполняется переиндексация")
                start = self.series.index.min()
                end = self.series.index.max()
                new_index = pd.date_range(start=start, end=end, freq='h')
                self.series = self.series.reindex(new_index).interpolate(method='linear')
                logging.info(f"Частота после переиндексации: {pd.infer_freq(self.series.index)}")

            self.series_index = self.series.index


            min_points = max(24, self.config['preprocessing'].get('lookback', 168) + self.horizon)
            if len(self.series) < min_points:
                raise ValueError(f"Недостаточно данных после обновления: {len(self.series)} точек, требуется минимум {min_points}")


            diff_series, d = self.analyzer.make_stationary(self.series, max_diff=1, name="Updated Series")
            self.series = diff_series
            self.series_index = self.series.index
            logging.info(f"Стационарный ряд обновлён с дифференцированием порядка {d}")
            logging.info(f"Частота после make_stationary для обновления: {pd.infer_freq(self.series.index)}")


            best_aic = float('inf')
            best_order = None
            best_seasonal_order = None

            combinations = [
                ((0, 1, 2), (1, 0, 0, 24)),
                ((1, 1, 2), (1, 0, 0, 24)),
                ((2, 1, 1), (1, 0, 1, 24))
            ]

            for order, seasonal_order in combinations:
                try:
                    model = ARIMA(self.series, order=order, seasonal_order=seasonal_order)
                    model_fit = model.fit()
                    aic = model_fit.aic
                    logging.info(f"Порядок {order}, сезонный порядок {seasonal_order}, AIC={aic:.4f}")
                    if aic < best_aic:
                        best_aic = aic
                        best_order = order
                        best_seasonal_order = seasonal_order
                except Exception as e:
                    logging.warning(f"Ошибка при обучении с {order}, {seasonal_order}: {e}")
                    continue

            if best_order is None:
                raise ValueError("Не удалось найти подходящую комбинацию параметров")

            self.order = best_order
            self.seasonal_order = best_seasonal_order
            self.model = ARIMA(self.series, order=self.order, seasonal_order=self.seasonal_order)
            self.model_fit = self.model.fit()
            logging.info(f"Выбран лучший порядок: order={self.order}, seasonal_order={self.seasonal_order}, AIC={best_aic:.4f}")

        except Exception as e:
            logging.error(f"Ошибка при обновлении ARIMA: {e}")
            raise ValueError(f"Не удалось обновить модель ARIMA: {e}")

    def forecast(self, horizon):
        try:
            if self.model_fit is None:
                raise ValueError("Модель не обучена.")

            if self.series_index is None or len(self.series_index) == 0:
                raise ValueError("Индекс временного ряда не инициализирован")


            inferred_freq = pd.infer_freq(self.series_index)
            logging.info(f"Частота индекса перед прогнозом: {inferred_freq}")
            if inferred_freq not in ['H', 'h']:
                raise ValueError(f"Индекс временного ряда не имеет почасовой частоты (обнаружено: {inferred_freq})")

            logging.info(f"Создание прогноза ARIMA на горизонт {horizon}")
            forecast_result = self.model_fit.forecast(steps=horizon)
            forecast = np.array(forecast_result)

            conf_int = self.model_fit.get_forecast(steps=horizon).conf_int(alpha=0.05)
            conf_int_df = pd.DataFrame({
                'lower': conf_int['lower ' + self.series.name],
                'upper': conf_int['upper ' + self.series.name]
            })

            if len(forecast) != horizon:
                logging.warning(f"Длина прогноза ({len(forecast)}) не соответствует горизонту ({horizon}), корректируется")
                if len(forecast) > horizon:
                    forecast = forecast[:horizon]
                    conf_int_df = conf_int_df.iloc[:horizon]
                else:
                    forecast = np.pad(forecast, (0, horizon - len(forecast)), mode='constant', constant_values=np.nan)
                    conf_int_df = conf_int_df.reindex(range(horizon), fill_value=np.nan)

            logging.info(f"Прогноз ARIMA успешно создан, длина={len(forecast)}")
            return forecast, conf_int_df

        except Exception as e:
            logging.error(f"Ошибка прогнозирования ARIMA: {e}")
            raise ValueError(f"Не удалось выполнить прогноз ARIMA: {e}")