import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class AnomalyFix:
    def __init__(self, config):
        """
        Инициализация класса для обработки аномалий с параметрами из конфигурации
        """
        self.method = config['preprocessing']['anom_method']  # 'rf' или 'stl'
        self.action = config['preprocessing']['anom_act']    # 'interpolate', 'remove' или 'mixed'
        self.period = config['preprocessing']['period']      # 24
        self.k = config['preprocessing']['k']                # 4
        self.thresh = config['preprocessing']['thresh']      # 4.0
        self.n_estimators = config['preprocessing']['rf_n_estimators']  # 150
        self.scaler = StandardScaler()

    def _linear_interpolation(self, series, indices):
        interpolated = series.copy()
        for idx in indices:
            prev_idx = next((i for i in range(idx-1, -1, -1) if not np.isnan(series[i]) and i not in indices), None)
            next_idx = next((i for i in range(idx+1, len(series)) if not np.isnan(series[i]) and i not in indices), None)
            if prev_idx is not None and next_idx is not None:
                y_prev, y_next = series[prev_idx], series[next_idx]
                t_prev, t_next = prev_idx, next_idx
                interpolated[idx] = y_prev + (y_next - y_prev) * (idx - t_prev) / (t_next - t_prev)
            else:
                interpolated[idx] = np.nan
        return interpolated

    def _stl_anomaly_detection(self, series):
        try:
            stl = STL(series.dropna(), period=self.period, robust=True)
            result = stl.fit()
            residuals = result.resid
            sigma_R = np.std(residuals)
            threshold = self.k * sigma_R
            anomalies = np.abs(residuals) > threshold
            anomaly_indices = series.index[anomalies]
            return anomaly_indices
        except Exception as e:
            logging.error(f"Ошибка в STL-разложении {str(e)}")
            return []

    def _rf_anomaly_detection(self, df, column, group, time_column):
        anomaly_indices = []
        for group_id in df[group].unique():
            group_data = df[df[group] == group_id].copy()
            if len(group_data) < self.period:
                logging.warning(f"Недостаточно данных для группы {group_id}. Пропуск.")
                continue

            group_data['lag1'] = group_data[column].shift(1)
            group_data['lag2'] = group_data[column].shift(2)
            group_data['lag24'] = group_data[column].shift(24)
            group_data['lag48'] = group_data[column].shift(48)  # Новый лаг
            group_data['rolling_mean'] = group_data[column].rolling(window=24, min_periods=1).mean()
            group_data['rolling_std'] = group_data[column].rolling(window=24, min_periods=1).std()
            group_data['hour'] = group_data[time_column].dt.hour
            group_data['day_of_week'] = group_data[time_column].dt.dayofweek

            feature_cols = ['lag1', 'lag2', 'lag24', 'lag48', 'rolling_mean', 'rolling_std', 'hour', 'day_of_week']
            valid_data = group_data.dropna(subset=feature_cols + [column])

            if len(valid_data) < self.n_estimators // 10:
                logging.warning(f"Недостаточно данных после удаления пропусков для группы {group_id}")
                continue

            X = valid_data[feature_cols]
            y = valid_data[column]
            X_scaled = self.scaler.fit_transform(X)

            rf = RandomForestRegressor(n_estimators=self.n_estimators, random_state=42)
            rf.fit(X_scaled, y)

            y_pred = rf.predict(X_scaled)
            errors = np.abs(y - y_pred)
            mad = np.median(errors)
            threshold = self.thresh * mad
            anomaly_mask = errors > threshold
            anomaly_indices.extend(valid_data.index[anomaly_mask])

        return anomaly_indices

    def fix(self, df, column, group='uuid', time_column='time_dt'):
        if column not in df.columns or group not in df.columns or time_column not in df.columns:
            raise ValueError("Нет необходимых столбцов")

        df_copy = df.copy()
        logging.info(f"Начало обработки аномалий методом {self.method}")

        if self.method == 'stl':
            for group_id in df_copy[group].unique():
                group_data = df_copy[df_copy[group] == group_id]
                if group_data[column].isna().all():
                    continue
                anomaly_indices = self._stl_anomaly_detection(group_data[column])
                if anomaly_indices.empty:
                    continue
                if self.action == 'interpolate' or (self.action == 'mixed' and not any(group_data[column][anomaly_indices] > 1500)):
                    series = df_copy.loc[df_copy[group] == group_id, column].values
                    indices = [df_copy.loc[df_copy[group] == group_id].index.get_loc(idx) for idx in anomaly_indices]
                    interpolated = self._linear_interpolation(series, indices)
                    df_copy.loc[df_copy[group] == group_id, column] = interpolated
                elif self.action == 'remove' or (self.action == 'mixed' and any(group_data[column][anomaly_indices] > 1500)):
                    df_copy.loc[anomaly_indices, column] = np.nan
        else:  # rf
            anomaly_indices = self._rf_anomaly_detection(df_copy, column, group, time_column)
            if anomaly_indices:
                if self.action == 'interpolate' or (self.action == 'mixed' and not any(df_copy.loc[anomaly_indices, column] > 1500)):
                    for group_id in df_copy[group].unique():
                        group_mask = df_copy[group] == group_id
                        group_indices = [idx for idx in anomaly_indices if idx in df_copy[group_mask].index]
                        if group_indices:
                            series = df_copy.loc[group_mask, column].values
                            indices = [df_copy.loc[group_mask].index.get_loc(idx) for idx in group_indices]
                            interpolated = self._linear_interpolation(series, indices)
                            df_copy.loc[group_mask, column] = interpolated
                elif self.action == 'remove' or (self.action == 'mixed' and any(df_copy.loc[anomaly_indices, column] > 1500)):
                    df_copy.loc[anomaly_indices, column] = np.nan

        if self.action in ['remove', 'mixed']:
            df_copy[column] = df_copy.groupby(group)[column].transform(
                lambda x: x.interpolate(method='linear').bfill().ffill())

        original_data = df[df[column].notna()][column]
        processed_data = df_copy[df_copy[column].notna()][column]
        if len(original_data) > 0 and len(processed_data) > 0:
            min_len = min(len(original_data), len(processed_data))
            mae = np.mean(np.abs(original_data[:min_len] - processed_data[:min_len]))
            rmse = np.sqrt(np.mean((original_data[:min_len] - processed_data[:min_len])**2))
            logging.info(f"MAE после обработки аномалий: {mae:.4f}")
            logging.info(f"RMSE после обработки аномалий: {rmse:.4f}")

        logging.info(f"Обработка аномалий завершена. Обнаружено аномалий: {len(anomaly_indices)}")
        return df_copy