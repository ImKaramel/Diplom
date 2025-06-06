import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from src.preprocessing.norm_fix import NormFix

class XGBoostModel:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.train_data = None
        self.booster = None
        self.lookback = config['preprocessing'].get('period', 24) * 7
        self.horizon = config['forecasting'].get('horizon', 168)
        self.num_boost_round = config['forecasting'].get('xgboost', {}).get('num_boost_round', 400)
        self.params = {
            'max_depth': config['forecasting'].get('xgboost', {}).get('max_depth', 8),
            'learning_rate': config['forecasting'].get('xgboost', {}).get('learning_rate', 0.1),
            'gamma': config['forecasting'].get('xgboost', {}).get('gamma', 0.1),
            'lambda': config['forecasting'].get('xgboost', {}).get('lambda', 2),
            'subsample': config['forecasting'].get('xgboost', {}).get('subsample', 0.8),
            'colsample_bytree': config['forecasting'].get('xgboost', {}).get('colsample_bytree', 0.8),
            'min_child_weight': config['forecasting'].get('xgboost', {}).get('min_child_weight', 1),
            'objective': 'reg:squarederror',
            'random_state': 42
        }
        self.early_stopping_rounds = config['forecasting'].get('xgboost', {}).get('early_stopping_rounds', 10)
        self.norm = NormFix(
            method=config.get('preprocessing', {}).get('norm_method', 'minmax'),
            params_file=config.get('data', {}).get('norm_params', 'data/processed/norm_params.json')
        )
        logging.info(f"Инициализированы параметры XGBoost -  {self.params}")
        logging.info(f"Количество итераций (num_boost_round): {self.num_boost_round}")

    def _create_features(self, data):
        df = data.copy()
        df['time_dt'] = pd.to_datetime(df['time_dt'])

        for lag in [24, 48, 168]:
            df[f'lag_{lag}'] = df['target'].shift(lag).astype(float)

        df['hour'] = df['time_dt'].dt.hour.astype(int)
        df['day_of_week'] = df['time_dt'].dt.dayofweek.astype(int)
        df['month'] = df['time_dt'].dt.month.astype(int)
        df['is_weekend'] = df['time_dt'].dt.dayofweek.isin([5, 6]).astype(int)

        if 'group' not in df.columns:
            df['group'] = 'default_group'

        df = pd.get_dummies(df, columns=['group'], drop_first=False, dtype=int)

        for lag in [24, 48, 168]:
            df[f'lag_{lag}'] = df[f'lag_{lag}'].fillna(df['target'].mean()).astype(float)

        if df.isna().any().any():
            logging.warning("В данных есть пропуски после заполнения лагов!!!, заполняю средними значениями")
            df = df.fillna(df.mean(numeric_only=True))

        return df

    def train(self, train_data):
        self.train_data = train_data.copy()
        features_df = self._create_features(train_data)

        if len(features_df) < self.lookback:
            logging.warning(f"Недостаточно данных для обучения ({len(features_df)} вместо {self.lookback}), качество может пострадать")
        elif len(features_df) == 0:
            raise ValueError("Нет данных для обучения после предобработки")

        X = features_df.drop(columns=['time_dt', 'target'])
        y = features_df['target']

        #  тренировочные и валидационные данные - поппытка снизить переобучение
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)


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
        logging.info(f"Модель XGBoost обучена для группы {train_data['group'].iloc[0] if 'group' in train_data.columns else 'default_group'}")
        logging.info(f"Лучшая итерация: {self.booster.best_iteration}")

    def forecast(self, horizon):
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала выполните обучение.")
        if self.train_data is None:
            raise ValueError("Нет тренировочных данных для прогнозирования.")

        last_data = self._create_features(self.train_data.tail(self.lookback))
        if len(last_data) == 0:
            logging.error("Нет данных для прогнозирования после предобработки")
            return np.zeros(horizon), None
        if len(last_data) < self.lookback:
            logging.warning(f"Недостаточно данных для lookback ({len(last_data)} вместо {self.lookback})")
        if len(last_data) < horizon:
            logging.warning(f"Недостаточно данных для прогноза ({len(last_data)} вместо {horizon})")

        forecast_vals = []
        current_data = last_data.copy()

        for lag in [24, 48, 168]:
            current_data[f'lag_{lag}'] = current_data[f'lag_{lag}'].astype(float)
        current_data['hour'] = current_data['hour'].astype(int)
        current_data['day_of_week'] = current_data['day_of_week'].astype(int)
        current_data['month'] = current_data['month'].astype(int)
        current_data['is_weekend'] = current_data['is_weekend'].astype(int)

        group_columns = [col for col in current_data.columns if col.startswith('group_')]
        if not group_columns:
            logging.warning("нужно проверить данные и логи")
        else:
            for col in group_columns:
                current_data[col] = current_data[col].astype(int)

        for i in range(horizon):
            X = current_data.drop(columns=['time_dt', 'target']).tail(1)
            if len(X) == 0:
                logging.warning(f"Нет данных для предсказания на шаге {i}, возвращаются последние предсказанные значения")
                if forecast_vals:
                    forecast_vals.extend([forecast_vals[-1]] * (horizon - i))
                else:
                    forecast_vals.extend([0] * (horizon - i))
                break

            dtest = xgb.DMatrix(X)
            pred = self.model.predict(dtest)[0]
            forecast_vals.append(pred)

            new_row = current_data.iloc[-1].copy()
            new_row['time_dt'] = new_row['time_dt'] + pd.Timedelta(hours=1)
            new_row['target'] = float(pred)
            for lag in [24, 48, 168]:
                lag_value = forecast_vals[-1] if len(forecast_vals) >= 1 else current_data['target'].iloc[-lag] if len(current_data) > lag else current_data['target'].mean()
                new_row[f'lag_{lag}'] = float(lag_value)
            new_row['hour'] = int(new_row['time_dt'].hour)
            new_row['day_of_week'] = int(new_row['time_dt'].dayofweek)
            new_row['month'] = int(new_row['time_dt'].month)
            new_row['is_weekend'] = 1 if new_row['time_dt'].dayofweek in [5, 6] else 0

            new_row_dict = new_row.to_dict()
            for col in group_columns:
                if col not in new_row_dict:
                    new_row_dict[col] = int(current_data[col].iloc[-1])
                else:
                    new_row_dict[col] = int(new_row_dict[col])
            new_row = pd.Series(new_row_dict)

            current_data = pd.concat([current_data, new_row.to_frame().T], ignore_index=True)

            for lag in [24, 48, 168]:
                current_data[f'lag_{lag}'] = current_data[f'lag_{lag}'].astype(float)
            current_data['hour'] = current_data['hour'].astype(int)
            current_data['day_of_week'] = current_data['day_of_week'].astype(int)
            current_data['month'] = current_data['month'].astype(int)
            current_data['is_weekend'] = current_data['is_weekend'].astype(int)
            for col in group_columns:
                current_data[col] = current_data[col].astype(int)

        return np.array(forecast_vals[:horizon]), None

    def evaluate(self, actual, forecast):
        mae = mean_absolute_error(actual, forecast)
        rmse = np.sqrt(mean_squared_error(actual, forecast))
        return mae, rmse