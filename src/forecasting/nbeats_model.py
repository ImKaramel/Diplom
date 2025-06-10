import logging
import numpy as np
import pandas as pd
import torch.nn as nn
from darts import TimeSeries
from darts.models import NBEATSModel
from sklearn.metrics import mean_absolute_error, mean_squared_error

class NBEATSInitializer:
    def __init__(self, config):
        self.input_chunk_length = config['preprocessing'].get('lookback', config['preprocessing'].get('period', 24) * 7)
        self.output_chunk_length = config['forecasting'].get('horizon', 168)
        self.num_stacks = config['forecasting'].get('nbeats', {}).get('num_stacks', 2)
        self.num_blocks = config['forecasting'].get('nbeats', {}).get('num_blocks', 3)
        self.num_layers = config['forecasting'].get('nbeats', {}).get('num_layers', 4)
        self.layer_widths = config['forecasting'].get('nbeats', {}).get('layer_widths', 512)
        self.n_epochs = config['forecasting'].get('nbeats', {}).get('n_epochs', 20)

        if self.input_chunk_length <= 0 or self.output_chunk_length <= 0:
            raise ValueError("input_chunk_length и output_chunk_length должны быть положительными")

        logging.info(f"Инициализация N-BEATS: input_chunk_length={self.input_chunk_length}, "
                     f"output_chunk_length={self.output_chunk_length}, n_epochs={self.n_epochs}")

    def initialize_model(self):
        try:
            model = NBEATSWrapper(
                input_chunk_length=self.input_chunk_length,
                output_chunk_length=self.output_chunk_length,
                num_stacks=self.num_stacks,
                num_blocks=self.num_blocks,
                num_layers=self.num_layers,
                layer_widths=self.layer_widths,
                n_epochs=self.n_epochs
            )
            logging.info(f"Модель N-BEATS инициализирована с precision=32-true, horizon={self.output_chunk_length}")
            return model
        except Exception as e:
            logging.error(f"Ошибка инициализации модели N-BEATS: {str(e)}")
            raise ValueError(f"Не удалось инициализировать модель N-BEATS: {str(e)}")

class NBEATSWrapper:
    def __init__(self, input_chunk_length, output_chunk_length, num_stacks, num_blocks,
                 num_layers, layer_widths, n_epochs):
        self.model = NBEATSModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            generic_architecture=True,
            num_stacks=num_stacks,
            num_blocks=num_blocks,
            num_layers=num_layers,
            layer_widths=layer_widths,
            n_epochs=n_epochs,
            random_state=42,
            loss_fn=nn.MSELoss(),
            pl_trainer_kwargs={"precision": "32-true", "accelerator": "auto"}
        )
        self.horizon = output_chunk_length
        self.series_index = None
        self.group = None

    def train(self, train_data):
        try:
            if train_data['group'].nunique() != 1:
                raise ValueError("train_data должно содержать только одну группу (uuid)")
            self.group = train_data['group'].iloc[0]

            if train_data['time_dt'].duplicated().any():
                logging.warning(f"Обнаружены дубликаты в time_dt для группы {self.group}, удаляю с усреднением target")
                train_data = train_data.groupby(['group', 'time_dt'])['target'].mean().reset_index()

            min_points = self.model.input_chunk_length + self.horizon
            if len(train_data) < min_points:
                raise ValueError(f"Недостаточно данных для обучения: {len(train_data)} < {min_points}")

            train_data['time_dt'] = pd.to_datetime(train_data['time_dt'])
            inferred_freq = pd.infer_freq(train_data.sort_values('time_dt')['time_dt'])
            if inferred_freq != 'h':
                logging.warning(f"Частота данных не почасовая ({inferred_freq}) для группы {self.group}, устанавливаю 'h'")
                train_data = train_data.set_index('time_dt').resample('h').agg({
                    'target': 'mean',
                    'group': 'first'
                }).reset_index()

            logging.info(f"Обучение N-BEATS: {len(train_data)} точек для группы {self.group}")
            series = TimeSeries.from_dataframe(
                train_data,
                time_col='time_dt',
                value_cols='target',
                fill_missing_dates=True,
                freq='h'
            ).astype(np.float32)
            self.series_index = series.time_index

            self.model.fit(series)
            logging.info(f"Модель N-BEATS успешно обучена для группы {self.group}")
        except Exception as e:
            logging.error(f"Ошибка обучения модели N-BEATS для группы {self.group}: {str(e)}")
            raise ValueError(f"Не удалось обучить модель N-BEATS: {str(e)}")

    def forecast(self, horizon):
        try:
            logging.info(f"Создание прогноза N-BEATS на горизонт {horizon} для группы {self.group}")
            if horizon > self.horizon:
                logging.warning(f"Горизонт {horizon} превышает output_chunk_length {self.horizon}, используем рекурсивный прогноз")
                forecast_vals = self._recursive_forecast(horizon)
            else:
                forecast = self.model.predict(n=horizon)
                forecast_vals = forecast.values().flatten()

            logging.info(f"Прогноз N-BEATS успешно создан, длина={len(forecast_vals)} для группы {self.group}")
            return forecast_vals, None
        except Exception as e:
            logging.error(f"Ошибка прогнозирования N-BEATS для группы {self.group}: {str(e)}")
            raise ValueError(f"Не удалось выполнить прогноз: {horizon}: {str(e)}")

    def _recursive_forecast(self, horizon):
        forecast_vals = np.array([], dtype=np.float32)
        current_series = None
        steps_left = horizon

        while steps_left > 1:
            pred_horizon = min(self.horizon, steps_left)
            if current_series is None:
                forecast = self.model.predict(n=pred_horizon)
            else:
                forecast = self.model.predict(n=pred_horizon, series=current_series)
            forecast_vals = np.append(forecast_vals, forecast.values().flatten().astype(np.float32))
            steps_left -= pred_horizon
            if steps_left > 1:
                current_series = TimeSeries.from_values(
                    values=forecast_vals[-self.model.input_chunk_length:].astype(np.float32)
                )

        return forecast_vals[:horizon]