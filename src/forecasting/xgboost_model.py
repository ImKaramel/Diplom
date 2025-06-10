import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from src.preprocessing.norm_fix import NormFix

class XGBoostModel:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.train_data = None
        self.booster = None
        self.lookback = config['preprocessing'].get('lookback', 168)
        self.horizon = config['forecasting'].get('horizon', 168)
        self.num_boost_round = config['forecasting'].get('xgboost', {}).get('num_boost_round', 300)
        self.params = {
            'max_depth': config['forecasting'].get('xgboost', {}).get('max_depth', 10),
            'learning_rate': config['forecasting'].get('xgboost', {}).get('learning_rate', 0.03),
            'gamma': config['forecasting'].get('xgboost', {}).get('gamma', 0.005),
            'lambda': config['forecasting'].get('xgboost', {}).get('lambda', 3),
            'subsample': config['forecasting'].get('xgboost', {}).get('subsample', 0.87),
            'colsample_bytree': config['forecasting'].get('xgboost', {}).get('colsample_bytree', 0.85),
            'min_child_weight': config['forecasting'].get('xgboost', {}).get('min_child_weight', 3),
            'objective': 'reg:squarederror',
            'random_state': 42
        }
        self.early_stopping_rounds = config['forecasting'].get('xgboost', {}).get('early_stopping_rounds', 20)
        self.norm = NormFix(
            method=config.get('preprocessing', {}).get('norm_method', 'log'),
            params_file=config.get('data', {}).get('norm_params', 'data/processed/norm_params.json')
        )
        logging.info(f"Инициализированы параметры XGBoost - {self.params}")
        logging.info(f"Лаги: {[24, 48, 168]}")
        logging.info(f"Количество итераций (num_boost_round): {self.num_boost_round}")

    def _create_features(self, data):
        df = data.copy()
        df['time_dt'] = pd.to_datetime(df['time_dt'])

        for lag in [24, 48, 168]:
            df[f'lag_{lag}'] = df['target'].shift(lag).bfill().astype(float)

        df['hour'] = df['time_dt'].dt.hour.astype(int)
        df['day_of_week'] = df['time_dt'].dt.dayofweek.astype(int)
        df['month'] = df['time_dt'].dt.month.astype(int)
        df['is_weekend'] = df['time_dt'].dt.dayofweek.isin([5, 6]).astype(int)

        if 'group' not in df.columns:
            df['group'] = 'default_group'

        df = pd.get_dummies(df, columns=['group'], drop_first=False, dtype=int)

        if df.isna().any().any():
            logging.warning("Обнаружены пропуски в данных, заполняю средними значениями")
            df = df.fillna(df.mean(numeric_only=True))

        for col in df.columns:
            if col not in ['time_dt', 'group']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isna().any():
                    logging.warning(f"Столбец {col} содержит NaN после приведения типов, заполняю средним")
                    df[col] = df[col].fillna(df[col].mean())

        logging.info(f"Типы данных после обработки: {df.dtypes}")
        return df

    def train(self, train_data):
        self.train_data = train_data.copy()
        features_df = self._create_features(train_data)

        if len(features_df) < self.lookback:
            logging.warning(f"Недостаточно данных для обучения ({len(features_df)} вместо {self.lookback})")
        elif len(features_df) == 0:
            raise ValueError("Нет данных для обучения после предобработки")

        min_train = max(48, self.horizon * 2)
        if len(features_df) < min_train:
            raise ValueError(f"Недостаточно данных для обучения: {len(features_df)} точек, требуется минимум {min_train}")

        X = features_df.drop(columns=['time_dt', 'target'])
        y = features_df['target']

        val_size = min(self.horizon, len(features_df) // 5)
        X_train, X_val = X.iloc[:-val_size], X.iloc[-val_size:]
        y_train, y_val = y.iloc[:-val_size], y.iloc[-val_size:]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        self.booster = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False
        )
        self.model = self.booster
        logging.info(f"Модель XGBoost обучена для группы {train_data['group'].iloc[0]}")
        logging.info(f"Лучшая итерация: {self.booster.best_iteration}")

    def forecast(self, horizon, recursive_step=1, train_df=None):
        if self.model is None:
            raise ValueError("Модель не обучена")
        if train_df is None and self.train_data is None:
            raise ValueError("Нет тренировочных данных")


        data = train_df if train_df is not None else self.train_data
        last_data = self._create_features(data.tail(self.lookback).copy())
        if len(last_data) == 0:
            logging.error("Нет данных для прогнозирования")
            return np.zeros(horizon), None

        forecast_vals = []
        current_data = last_data.copy()

        numeric_cols = [col for col in current_data.columns if col not in ['time_dt', 'target', 'group']]

        for step in range(horizon):
            logging.info(f"Рекурсивный шаг {step + 1}/{horizon}")
            X = current_data.drop(columns=['time_dt', 'target']).tail(1)
            if len(X) == 0:
                logging.warning(f"Нет данных для предсказания на шаге {step}, заполняю нулями")
                forecast_vals.extend([0] * (horizon - step))
                break

            for col in numeric_cols:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(float)

            dtest = xgb.DMatrix(X)
            pred = self.model.predict(dtest)[0]
            forecast_vals.append(pred)

            new_row = current_data.iloc[-1].copy()
            new_row['time_dt'] = new_row['time_dt'] + pd.Timedelta(hours=recursive_step)
            new_row['target'] = float(pred)
            for lag in [24, 48, 168]:
                lag_value = forecast_vals[-1] if len(forecast_vals) >= lag else current_data['target'].iloc[-lag] if len(current_data) > lag else current_data['target'].mean()
                new_row[f'lag_{lag}'] = float(lag_value)
            new_row['hour'] = int(new_row['time_dt'].hour)
            new_row['day_of_week'] = int(new_row['time_dt'].dayofweek)
            new_row['month'] = int(new_row['time_dt'].month)
            new_row['is_weekend'] = 1 if new_row['time_dt'].dayofweek in [5, 6] else 0

            new_row_dict = new_row.to_dict()
            group_columns = [col for col in current_data.columns if col.startswith('group_')]
            for col in group_columns:
                new_row_dict[col] = int(current_data[col].iloc[-1])


            new_row = pd.Series(new_row_dict)
            new_row_numeric = new_row[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
            new_row = new_row.copy()
            for col in numeric_cols:
                new_row[col] = new_row_numeric[col]

            current_data = pd.concat([current_data, new_row.to_frame().T], ignore_index=True)

        return np.array(forecast_vals), None
