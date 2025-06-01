import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import RandomForestRegressor

class AnomalyFix:
    def __init__(self, config):
        self.method = config['preprocessing'].get('anom_method', 'iqr')  # iqr, stl, rf, mixed
        self.action = config['preprocessing'].get('anom_act', 'interpolate')  # interpolate, remove
        self.period = config['preprocessing'].get('period', 24)  # Суточный период
        self.k = config['preprocessing'].get('k', 4)  # Порог для STL
        self.thresh = config['preprocessing'].get('thresh', 4.0)  # Порог для RF
        self.n_estimators = config['preprocessing'].get('rf_n_estimators', 150)  # Кол-во деревьев для RF
        self.scaler = StandardScaler()
        self.anomaly_logger = logging.getLogger('anomaly_logger')
        self.anomaly_logger.setLevel(logging.INFO)
        anomaly_handler = logging.FileHandler('anomaly_log.log')
        anomaly_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.anomaly_logger.handlers = []
        self.anomaly_logger.addHandler(anomaly_handler)

    def _linear_interpolation(self, series, anomaly_positions, original_indices, df, column, group, group_id, logger):
        interpolated = series.copy()
        group_mean = df[df[group] == group_id][column].mean()
        fallback_mean = df[column].mean() if np.isnan(group_mean) else group_mean
        for pos in anomaly_positions:
            prev_idx = next((i for i in range(pos-1, -1, -1) if i not in anomaly_positions and not np.isnan(series[i])), None)
            next_idx = next((i for i in range(pos+1, len(series)) if i not in anomaly_positions and not np.isnan(series[i])), None)
            orig_idx = original_indices[anomaly_positions.index(pos)]
            if prev_idx is not None and next_idx is not None:
                y_prev, y_next = series[prev_idx], series[next_idx]
                interpolated[pos] = y_prev + (y_next - y_prev) * (pos - prev_idx) / (next_idx - prev_idx)
            else:
                interpolated[pos] = fallback_mean
                logger.warning(f"Интерполяция на индексе {orig_idx} невозможна, использовано среднее значение {fallback_mean:.2f}")
        return interpolated

    def _stl_anomaly_detection(self, series, logger):
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
            logger.error(f"Ошибка в STL-разложении: {str(e)}")
            self.anomaly_logger.error(f"Ошибка в STL-разложении: {str(e)}")
            return pd.Index([])

    def _rf_anomaly_detection(self, df, column, group, time_column, logger):
        anomaly_indices = []
        for group_id in df[group].unique():
            group_data = df[df[group] == group_id].copy()
            if len(group_data) < self.period:
                continue

            group_data['lag1'] = group_data[column].shift(1)
            group_data['lag24'] = group_data[column].shift(24)
            group_data['rolling_mean'] = group_data[column].rolling(window=24, min_periods=1).mean()
            group_data['hour'] = group_data[time_column].dt.hour

            feature_cols = ['lag1', 'lag24', 'rolling_mean', 'hour']
            valid_data = group_data.dropna(subset=feature_cols + [column])

            if len(valid_data) < self.n_estimators // 10:
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
            group_anomalies = valid_data.index[anomaly_mask]
            anomaly_indices.extend(group_anomalies)
        return pd.Index(anomaly_indices)

    def _iqr_anomaly_detection(self, df, column, group, logger):
        anomaly_indices = []
        for group_id in df[group].unique():
            group_data = df[df[group] == group_id][column].dropna()
            if len(group_data) < 10:
                continue
            Q1 = group_data.quantile(0.25)
            Q3 = group_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            anomaly_mask = (group_data < lower_bound) | (group_data > upper_bound)
            group_anomalies = group_data.index[anomaly_mask]
            anomaly_indices.extend(group_anomalies)
        return pd.Index(anomaly_indices)

    def _detect_anomalies(self, df, column, group, time_column):
        all_anomalies = pd.Index([])
        logger = logging.getLogger('anomaly_logger')

        if self.method == 'stl':
            for group_id in df[group].unique():
                group_data = df[df[group] == group_id][column].dropna()
                if len(group_data) < self.period:
                    continue
                stl_indices = self._stl_anomaly_detection(group_data, logger)
                all_anomalies = all_anomalies.union(stl_indices)
        elif self.method == 'rf':
            all_anomalies = self._rf_anomaly_detection(df, column, group, time_column, logger)
        elif self.method == 'iqr':
            all_anomalies = self._iqr_anomaly_detection(df, column, group, logger)
        else:  # mixed
            for group_id in df[group].unique():
                group_data = df[df[group] == group_id][column].dropna()
                if len(group_data) < self.period:
                    continue
                stl = STL(group_data, period=self.period, robust=True)
                result = stl.fit()
                seasonal_std = np.std(result.seasonal)
                if seasonal_std > 50:
                    stl_indices = self._stl_anomaly_detection(group_data, logger)
                    all_anomalies = all_anomalies.union(stl_indices)
                else:
                    rf_indices = self._rf_anomaly_detection(df[df[group] == group_id], column, group, time_column, logger)
                    all_anomalies = all_anomalies.union(rf_indices)
        return all_anomalies

    def fix(self, df, column, group='uuid', time_column='time_dt'):
        if column not in df.columns or group not in df.columns or time_column not in df.columns:
            raise ValueError("Отсутствуют необходимые столбцы")

        df_copy = df.copy()
        logger = logging.getLogger('anomaly_logger')
        logger.info(f"Начало обработки аномалий методом {self.method}")
        self.anomaly_logger.info(f"Начало обработки аномалий методом {self.method}")

        anomaly_indices = self._detect_anomalies(df_copy, column, group, time_column)

        if anomaly_indices.empty:
            logger.info("Аномалии не обнаружены")
            self.anomaly_logger.info("Аномалии не обнаружены")
            return df_copy

        anomaly_values = df_copy.loc[anomaly_indices, column].to_dict()
        #logger.info(f"Обнаружены аномалии: {anomaly_values}")
        #self.anomaly_logger.info(f"Обнаружены аномалии: {anomaly_values}")

        if self.action == 'interpolate':
            for group_id in df_copy[group].unique():
                mask = df_copy[group] == group_id
                group_data = df_copy.loc[mask].copy()
                group_indices = [idx for idx in anomaly_indices if idx in group_data.index]
                if group_indices:
                    series = group_data[column].values
                    anomaly_positions = [group_data.index.get_loc(idx) for idx in group_indices]
                    interpolated = self._linear_interpolation(series, anomaly_positions, group_indices, df_copy, column, group, group_id, logger)
                    for i, orig_idx in enumerate(group_indices):
                        pos = anomaly_positions[i]
                        if not np.isnan(interpolated[pos]):
                            df_copy.loc[orig_idx, column] = interpolated[pos]
        elif self.action == 'remove':
            df_copy.loc[anomaly_indices, column] = np.nan
            df_copy[column] = df_copy.groupby(group)[column].transform(
                lambda x: x.interpolate(method='linear').bfill().ffill())
            logger.info(f"Аномалии удалены и интерполированы для {len(anomaly_indices)} точек")
            self.anomaly_logger.info(f"Аномалии удалены и интерполированы для {len(anomaly_indices)} точек")

        plt.figure(figsize=(12, 6))
        plt.plot(df_copy[time_column], df_copy[column], label='Данные', color='blue')
        plt.scatter(df_copy.loc[anomaly_indices, time_column], df_copy.loc[anomaly_indices, column],
                    color='red', label='Аномалии', s=50)
        plt.title(f"Аномалии в данных группы {group}")
        plt.xlabel('Время')
        plt.ylabel('Потребление энергии (Вт·ч)')
        plt.legend()
        plt.grid(True)
        os.makedirs('graphics/anomalies', exist_ok=True)
        plot_path = f'graphics/anomalies/anomalies_{group}.png'
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"График аномалий сохранен: {plot_path}")
        self.anomaly_logger.info(f"График аномалий сохранен: {plot_path}")

        logger.info(f"Обработка завершена. Обнаружено аномалий: {len(anomaly_indices)}")
        self.anomaly_logger.info(f"Обработка завершена. Обнаружено аномалий: {len(anomaly_indices)}")
        return df_copy