import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from statsmodels.tsa.seasonal import STL


class AnomalyFix:
    def __init__(self, method='stl', action='interpolate', thresh=2.5, win=13, rf_n_estimators=100, lookback=24 * 7,
                 period=24):
        """
        - method: 'stl' (STL-разложение) или 'rf' (RandomForestRegressor)
        - action: 'interpolate' (линейная интерполяция) или 'remove' (удаление аномалий)
        - thresh: порог для определения аномалий (k в |R_t| > k * σ_R или |y_t - ŷ_t| > k * MAD)
        - win: размер окна для STL-сглаживания
        - rf_n_estimators: число деревьев в RandomForest
        - lookback: длина временного окна для признаков (лагов) в RandomForest
        - period: период сезонности (по умолчанию 24 часа)
        """
        self.method = method
        self.action = action
        self.thresh = thresh
        self.win = win
        self.rf_n_estimators = rf_n_estimators
        self.lookback = lookback
        self.period = period
        if method not in ['stl', 'rf']:
            raise ValueError("Метод: 'stl' или 'rf'")
        if action not in ['interpolate', 'remove']:
            raise ValueError("Действие: 'interpolate' или 'remove'")

    def _stl(self, data, period):
        """STL-разложение с использованием statsmodels"""
        try:
            data_clean = data.interpolate(method='linear').bfill().ffill()
            stl = STL(data_clean, period=period, robust=True)
            result = stl.fit()
            return {'trend': result.trend, 'season': result.seasonal, 'resid': result.resid}
        except Exception as e:
            logging.error(f"Ошибка STL-разложения: {str(e)}")
            raise

    def _sklearn_anomaly(self, data, time_column='time_dt', period=24, lookback=24 * 7, extra_features=None):
        try:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()

            df_features = pd.DataFrame({'value': data_scaled})
            if time_column in data.index.names:
                df_features['hour'] = data.index.get_level_values(time_column).hour
            else:
                df_features['hour'] = data[time_column].dt.hour

            for lag in range(1, min(lookback + 1, len(data))):
                df_features[f'lag_{lag}'] = data_scaled.shift(lag)

            if extra_features:
                valid_features = [f for f in extra_features if f in data.columns]
                for feature in valid_features:
                    df_features[feature] = scaler.fit_transform(data[feature].values.reshape(-1, 1)).flatten()

            df_features = df_features.dropna()
            if len(df_features) < 10:
                return np.array([False] * len(data)), "Недостаточно данных для RandomForest"

            X = df_features.drop(columns=['value']).values
            y = df_features['value'].values

            train_size = int(0.8 * len(X))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            model = RandomForestRegressor(n_estimators=self.rf_n_estimators, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X)
            errors = np.abs(y - y_pred)
            mad = np.median(errors)
            if mad == 0:
                return np.array([False] * len(data)), "MAD равно 0"

            anomaly = errors > self.thresh * mad
            anomaly_full = np.array([False] * len(data))
            anomaly_full[df_features.index] = anomaly

            return anomaly_full, None
        except Exception as e:
            return np.array([False] * len(data)), f"Ошибка RandomForest: {str(e)}"

    def _interp(self, data, mask):
        """Линейная интерполяция для замены аномалий"""
        out = data.copy()
        idx = np.where(mask)[0]
        for i in idx:
            prev = next((j for j in range(i - 1, -1, -1) if not mask[j] and not np.isnan(data[j])), None)
            next_i = next((j for j in range(i + 1, len(data)) if not mask[j] and not np.isnan(data[j])), None)
            if prev is not None and next_i is not None:
                y1, y2 = data[prev], data[next_i]
                t1, t2 = prev, next_i
                out[i] = y1 + (y2 - y1) * (i - t1) / (t2 - t1)
        return out

    def _clean(self, data):
        """Обработка пропусков с помощью линейной интерполяции (их не должно быть)"""
        if data.isna().any():
            data = data.interpolate(method='linear').bfill().ffill()
        return data

    def _check(self, data):
        if data.isna().any():
            return False, "Есть пропуски"
        if data.var() < 1e-10:
            return False, "Дисперсия почти 0"
        diff = data.diff().abs()
        if diff.max() > data.std() * 5:
            return False, f"Скачки: {diff.max():.2f} > {data.std():.2f}"
        return True, ""

    def _proc(self, group, df, col, groupby):
        logging.info(f"Обработка аномалий для {group} методом {self.method}")
        mask = df[groupby] == group
        data = df.loc[mask, col]
        result = {'mask': mask, 'anomaly': None, 'fixed': None, 'err': None}

        if len(data) > 2 * self.period and data.notna().sum() > 2 * self.period:
            try:
                clean = self._clean(data)
                ok, msg = self._check(clean)
                if not ok:
                    result['err'] = f"Пропуск: {msg}"
                    return result

                if self.method == 'stl':
                    stl = self._stl(clean, period=self.period)
                    res = stl['resid']
                    anomaly = np.abs(res) > self.thresh * np.std(res)
                else:  # rf
                    extra_features = ['R_plus', 'A_minus', 'R_minus'] if all(
                        f in df.columns for f in ['R_plus', 'A_minus', 'R_minus']) else None
                    anomaly, err = self._sklearn_anomaly(clean, time_column='time_dt', period=self.period,
                                                         lookback=self.lookback, extra_features=extra_features)
                    if err:
                        result['err'] = err
                        return result

                result['anomaly'] = anomaly
                if anomaly.any():
                    times = df.loc[mask, 'time_dt'][anomaly]
                    logging.info(f"Аномалий: {anomaly.sum()} для {group}")
                    for t, v in zip(times, data[anomaly]):
                        logging.info(f"Аномалия: {group}, {t}, A_plus={v}")

                if self.action == 'interpolate' and anomaly.any():
                    result['fixed'] = self._interp(data.values, anomaly)
                elif self.action == 'remove' and anomaly.any():
                    result['remove'] = ~anomaly
            except Exception as e:
                result['err'] = f"Ошибка обработки: {str(e)}"
        else:
            result['err'] = f"Мало данных: {len(data)}"

        logging.info(f"Размер для {group}: {len(data)}")
        return result

    def fix(self, df, col, groupby='uuid'):
        logging.info(f"Чищу аномалии методом {self.method}")
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df должен быть pandas.DataFrame")
        df_out = df.copy()
        df_out['anomaly'] = False
        total = 0

        try:
            with ThreadPoolExecutor() as ex:
                proc = partial(self._proc, df=df_out, col=col, groupby=groupby)
                results = list(ex.map(proc, df_out[groupby].unique()))
        except Exception as e:
            logging.error(f"Ошибка параллельной обработки: {str(e)}")
            raise

        for group, result in zip(df_out[groupby].unique(), results):
            if result['err']:
                logging.warning(f"{group}: {result['err']}")
                logging.info(f"NaN={df_out[df_out[groupby] == group][col].isna().sum()}, "
                             f"Inf={np.isinf(df_out[df_out[groupby] == group][col]).sum()}, "
                             f"Var={df_out[df_out[groupby] == group][col].var()}, "
                             f"Min={df_out[df_out[groupby] == group][col].min()}, "
                             f"Max={df_out[df_out[groupby] == group][col].max()}")
                continue

            mask = result['mask']
            if result['anomaly'] is not None:
                df_out.loc[mask, 'anomaly'] = result['anomaly']
                total += result['anomaly'].sum()

                if self.action == 'interpolate' and result['fixed'] is not None:
                    df_out.loc[mask, col] = result['fixed']
                    df_out.loc[mask, col] = df_out.loc[mask, col].fillna(
                        df_out.loc[mask, col].mean() if df_out.loc[mask, col].notna().any() else 0)
                elif self.action == 'remove' and 'remove' in result:
                    df_out = df_out.loc[~((df_out[groupby] == group) & ~result['remove'])]

        logging.info(f"Обнаружено аномалий: {total}")
        return df_out
