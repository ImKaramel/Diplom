import logging
import numpy as np
import torch.nn as nn
from darts import TimeSeries
from darts.models import NBEATSModel

class NBEATSInitializer:
    def __init__(self, config):
        self.input_chunk_length = config['preprocessing'].get('period', 24) * 7  # Ретроспектива
        self.output_chunk_length = config['forecasting'].get('horizon', 24)  # Горизонт прогноза
        self.num_stacks = config['forecasting'].get('nbeats', {}).get('num_stacks', 2)  # Количество стеков
        self.num_blocks = config['forecasting'].get('nbeats', {}).get('num_blocks', 3)  # Количество блоков в стеке
        self.num_layers = config['forecasting'].get('nbeats', {}).get('num_layers', 4)  # Количество слоев в блоке
        self.layer_widths = config['forecasting'].get('nbeats', {}).get('layer_widths', 512)  # Ширина слоев
        self.n_epochs = config['forecasting'].get('nbeats', {}).get('n_epochs', 50)  # Количество эпох

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
            logging.info("Модель N-BEATS инициализирована с precision=32-true для MPS")
            return model
        except Exception as e:
            logging.error(f"Ошибка инициализации модели N-BEATS: {str(e)}")
            raise

class NBEATSWrapper:
    def __init__(self, input_chunk_length, output_chunk_length, num_stacks, num_blocks, num_layers, layer_widths, n_epochs):
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
            pl_trainer_kwargs={"precision": "32-true"}
        )
        self.horizon = output_chunk_length

    def train(self, train_data):
        try:
            if 'time_dt' not in train_data or 'target' not in train_data:
                raise ValueError("Данные должны содержать столбцы 'time_dt' и 'target'")

            logging.info(f"Обучение N-BEATS: {len(train_data)} точек")
            series = TimeSeries.from_dataframe(
                train_data, time_col='time_dt', value_cols='target', fill_missing_dates=True, freq='h'
            ).astype(np.float32)
            self.model.fit(series)
            logging.info("Модель N-BEATS успешно обучена")
        except Exception as e:
            logging.error(f"Ошибка обучения модели N-BEATS: {str(e)}")
            raise ValueError(f"Не удалось обучить модель N-BEATS: {str(e)}")

    def forecast(self, horizon):
        try:
            if horizon != self.horizon:
                logging.warning(f"Запрошенный горизонт {horizon} отличается от output_chunk_length {self.horizon}, использую {self.horizon}")
                horizon = self.horizon
            logging.info(f"Создание прогноза N-BEATS на горизонт {horizon}")
            forecast = self.model.predict(n=horizon)
            forecast_vals = forecast.values().flatten()
            logging.info(f"Прогноз N-BEATS успешно создан, длина={len(forecast_vals)}")
            return forecast_vals, None  #  не возвращает доверительные интервалы
        except Exception as e:
            logging.error(f"Ошибка прогнозирования N-BEATS: {str(e)}")
            raise ValueError(f"Не удалось выполнить прогноз N-BEATS: {str(e)}")