import logging
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.preprocessing.norm_fix import NormFix

class XGBoostModel:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.train_data = None
        self.lookback = config['preprocessing'].get('period', 24) * 7
        self.horizon = config['forecasting'].get('horizon', 24)
        self.params = {
            'n_estimators': config['forecasting'].get('xgboost', {}).get('n_estimators', 100),
            'max_depth': config['forecasting'].get('xgboost', {}).get('max_depth', 6),
            'learning_rate': config['forecasting'].get('xgboost', {}).get('learning_rate', 0.1),
            'gamma': config['forecasting'].get('xgboost', {}).get('gamma', 0),
            'lambda': config['forecasting'].get('xgboost', {}).get('lambda', 1),
            'objective': 'reg:squarederror',
            'random_state': 42
        }
        self.norm = NormFix(
            method=config.get('preprocessing', {}).get('norm_method', 'minmax'),
            params_file=config.get('data', {}).get('norm_params', 'data/processed/norm_params.json')
        )

    def _create_features(self, data):
        df = data.copy()
        df['time_dt'] = pd.to_datetime(df['time_dt'])

        for lag in [24, 48]:
            df[f'lag_{lag}'] = df['target'].shift(lag)

        df['hour'] = df['time_dt'].dt.hour
        df['day_of_week'] = df['time_dt'].dt.dayofweek
        df['month'] = df['time_dt'].dt.month

        df = pd.get_dummies(df, columns=['group'], drop_first=True)
        df = df.dropna()

        return df

    def train(self, train_data):
        self.train_data = train_data.copy()
        features_df = self._create_features(train_data)

        X = features_df.drop(columns=['time_dt', 'target'])
        y = features_df['target']

        self.model = XGBRegressor(**self.params)
        self.model.fit(X, y)
        logging.info(f"Модель XGBoost обучена для группы {train_data['group'].iloc[0]}")

    def forecast(self, horizon):
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала выполните обучение.")
        if self.train_data is None:
            raise ValueError("Нет тренировочных данных для прогнозирования.")

        last_data = self._create_features(self.train_data.tail(self.lookback))
        if len(last_data) < horizon:
            logging.warning(f"Недостаточно данных для прогноза ({len(last_data)} вместо {horizon}), дополняю последними значениями")
            last_row = last_data.iloc[-1:].copy()
            while len(last_data) < horizon:
                last_row['time_dt'] += pd.Timedelta(hours=1)
                last_data = pd.concat([last_data, last_row], ignore_index=True)

        X = last_data.drop(columns=['time_dt', 'target'])

        # Прогноз
        forecast = self.model.predict(X[-horizon:])
        return forecast, None

    def evaluate(self, actual, forecast):
        mae = mean_absolute_error(actual, forecast)
        rmse = np.sqrt(mean_squared_error(actual, forecast))
        return mae, rmse