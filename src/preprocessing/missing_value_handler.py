import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

class MissingValueHandler:
    def __init__(self, method='linear', period=24, k=5):
        self.method = method
        self.period = period
        self.k = k
        if method not in ['linear', 'seasonal', 'knn']:
            raise ValueError("Метод должен быть 'linear', 'seasonal' или 'knn'")

    def _linear_interpolation(self, series, indices):
        interpolated = series.copy()
        for idx in indices:
            prev_idx = next((i for i in range(idx-1, -1, -1) if not np.isnan(series[i])), None)
            next_idx = next((i for i in range(idx+1, len(series)) if not np.isnan(series[i])), None)
            if prev_idx is not None and next_idx is not None:
                y_prev, y_next = series[prev_idx], series[next_idx]
                t_prev, t_next = prev_idx, next_idx
                interpolated[idx] = y_prev + (y_next - y_prev) * (idx - t_prev) / (t_next - t_prev)
            else:
                interpolated[idx] = np.nan
        return interpolated

    def _knn_imputation(self, df, column, groupby, time_column):
        df_copy = df.copy()
        for group in df_copy[groupby].unique():
            group_data = df_copy[df_copy[groupby] == group]
            mask = group_data[column].isna()
            if mask.any():
                valid_data = group_data[~mask]
                missing_data = group_data[mask]
                if len(valid_data) < self.k:
                    logging.warning(f"Недостаточно данных для KNN (uuid={group}) --- Заполнение средним")
                    df_copy.loc[mask, column] = valid_data[column].mean() if valid_data[column].notna().any() else 0
                    continue
                X_valid = (valid_data[time_column] - valid_data[time_column].min()).dt.total_seconds().values.reshape(-1, 1)
                X_missing = (missing_data[time_column] - valid_data[time_column].min()).dt.total_seconds().values.reshape(-1, 1)
                y_valid = valid_data[column].values
                knn = NearestNeighbors(n_neighbors=self.k, metric='euclidean')
                knn.fit(X_valid)
                distances, indices = knn.kneighbors(X_missing)
                for i, idx in enumerate(missing_data.index):
                    weights = 1 / (distances[i] + 1e-10)
                    weights /= weights.sum()
                    df_copy.loc[idx, column] = np.sum(weights * y_valid[indices[i]])
        return df_copy

    def _seasonal_interpolation(self, df, column, groupby, time_column):
        df_copy = df.copy()
        for group in df_copy[groupby].unique():
            group_mask = df_copy[groupby] == group
            missing_indices = df_copy[group_mask & df_copy[column].isna()].index
            for idx in missing_indices:
                target_time = df_copy.loc[idx, time_column]
                seasonal_values = []
                for i in range(1, 10):
                    past_time = target_time - pd.Timedelta(hours=self.period * i)
                    past_value = df_copy.loc[(df_copy[groupby] == group) &
                                             (df_copy[time_column] == past_time), column]
                    if past_value.notna().any():
                        seasonal_values.append(past_value.values[0])
                if seasonal_values:
                    df_copy.loc[idx, column] = np.mean(seasonal_values)
                else:
                    logging.warning(f"Нет данных для сезонной интерполяции в uuid={group}, time={target_time}")
                    df_copy.loc[idx, column] = df_copy.loc[group_mask, column].mean() if df_copy.loc[group_mask, column].notna().any() else 0
        return df_copy

    def handle(self, df, column, groupby='uuid', time_column='time_dt'):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df должен быть pandas.DataFrame")
        if column not in df.columns or time_column not in df.columns or groupby not in df.columns:
            raise ValueError(f"Нет нужных столбцов: {column}, {time_column}, {groupby}")

        df_copy = df.reset_index(drop=True) if time_column in df.columns and df.index.name == time_column else df.copy()
        if time_column not in df_copy.columns and df.index.name == time_column:
            df_copy[time_column] = df.index

        df_copy[time_column] = pd.to_datetime(df_copy[time_column])

        all_data = []
        for group in df_copy[groupby].unique():
            group_data = df_copy[df_copy[groupby] == group]
            if group_data[time_column].isna().any():
                logging.warning(f"Обнаружены пропуски в {time_column} для uuid={group}")
                continue
            min_time = group_data[time_column].min()
            max_time = group_data[time_column].max()
            if pd.isna(min_time) or pd.isna(max_time):
                logging.warning(f"Некорректные временные метки для uuid={group}")
                continue
            full_time_index = pd.date_range(start=min_time, end=max_time, freq='h')
            full_df = pd.DataFrame({time_column: full_time_index, groupby: group})
            group_data = group_data.reset_index(drop=True)
            merged_df = full_df.merge(group_data, on=[groupby, time_column], how='left')
            all_data.append(merged_df)

        if not all_data:
            raise ValueError("Нет данных для обработки")

        df_copy = pd.concat(all_data, ignore_index=True)
        missing_count = df_copy[column].isna().sum()
        logging.info(f"Обнаружено {missing_count} пропусков (отсутствующих строк) в {column}")

        if missing_count > 0:
            if self.method == 'seasonal':
                df_copy = self._seasonal_interpolation(df_copy, column, groupby, time_column)
                df_copy[column] = df_copy.groupby(groupby)[column].transform(
                    lambda x: x.interpolate(method='linear').bfill().ffill())
            elif self.method == 'knn':
                df_copy = self._knn_imputation(df_copy, column, groupby, time_column)
            else:  # linear
                for group in df_copy[groupby].unique():
                    group_mask = df_copy[groupby] == group
                    series = df_copy.loc[group_mask, column].values
                    indices = df_copy.loc[group_mask & df_copy[column].isna()].index
                    if len(indices) > 0:
                        interpolated = self._linear_interpolation(series, indices)
                        df_copy.loc[group_mask, column] = interpolated
                df_copy[column] = df_copy.groupby(groupby)[column].transform(
                    lambda x: x.fillna(x.mean() if x.notna().any() else 0))

        for col in ['A_minus', 'R_plus', 'R_minus']:
            if col in df_copy.columns:
                df_copy[col] = df_copy.groupby(groupby)[col].transform(
                    lambda x: x.fillna(x.median() if x.notna().any() else x.min() if x.notna().any() else 0))

        if missing_count > 0:
            variance = df_copy[column].var()
            logging.info(f"Дисперсия ряда: {variance:.2f}")

        remaining_missing = df_copy[column].isna().sum()
        logging.info(f"Обнаружено {remaining_missing} пропусков в {column} после заполнения")
        return df_copy