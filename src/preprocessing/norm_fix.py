import logging
import json
import numpy as np
import pandas as pd
import os

class NormFix:
    def __init__(self, method='minmax', params_file='data/processed/norm_params.json'):
        """
            - minmax: (y - y_min) / (y_max - y_min)
            - log: log(y + 1)
            - user_minmax: MinMax-нормализация по группам
        """
        self.method = method
        self.params_file = params_file
        self.params = {}
        if method not in ['minmax', 'log', 'user_minmax']:
            raise ValueError("Метод: 'minmax', 'log' или 'user_minmax'")

    def _check(self, df, col, group):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df должен быть pandas.DataFrame")
        if df[col].isna().any():
            logging.warning(f"Пропуски в {col}: {df[col].isna().sum()}")

    def _handle_missing(self, series):
        if series.isna().any():
            median = series.median()
            if pd.isna(median):
                logging.warning("Медиана не определена Использую интерполяцию")
                return series.interpolate(method='linear', limit_direction='both'), None
            return series.fillna(median), median
        return series, None

    def _save_params(self):
        try:
            os.makedirs(os.path.dirname(self.params_file), exist_ok=True)
            with open(self.params_file, 'w') as f:
                json.dump(self.params, f, indent=4)
            logging.info(f"Параметры нормализации сохранены в {self.params_file}")
        except Exception as e:
            logging.error(f"Ошибка {str(e)}")
            raise

    def fix(self, df, col, group='uuid'):
        df_out = df.copy()
        self._check(df_out, col, group)

        logging.info(f"До нормализации: min={df_out[col].min():.4f}, max={df_out[col].max():.4f}, mean={df_out[col].mean():.4f}")

        if self.method == 'log':
            series, _ = self._handle_missing(df_out[col])
        else:
            series, median = self._handle_missing(df_out[col])
            df_out[col] = series

        if self.method == 'log':
            if (df_out[col] < 0).any():
                logging.error("Обнаружены отрицательные значения в A+. Ошибка для log (не должно быть так)")
                raise ValueError("Отрицательные значения не допускаются для log ")
            df_out[col] = np.log(df_out[col] + 1)
            self.params = {'method': 'log'}
        elif self.method == 'user_minmax':
            def norm_group(x):
                if x.notna().sum() < 2:
                    logging.warning(f"Мало данных для группы {x.name}")
                    return np.zeros_like(x), 0, 0
                x_min, x_max = x.min(), x.max()
                if x_max == x_min:
                    logging.warning(f"Диапазон 0 для группы {x.name}")
                    return np.zeros_like(x), x_min, x_max
                return (x - x_min) / (x_max - x_min), x_min, x_max

            self.params = {'method': 'user_minmax', 'groups': {}}
            for g in df_out[group].unique():
                mask = df_out[group] == g
                df_out.loc[mask, col], min_val, max_val = norm_group(df_out.loc[mask, col])
                self.params['groups'][str(g)] = {'min': float(min_val), 'max': float(max_val)}
        else:
            x_min, x_max = df_out[col].min(), df_out[col].max()
            if x_max == x_min:
                logging.warning("Диапазон 0. Возвращаю нули.")
                df_out[col] = np.zeros_like(df_out[col])
                self.params = {'method': 'minmax', 'min': float(x_min), 'max': float(x_max)}
            else:
                df_out[col] = (df_out[col] - x_min) / (x_max - x_min)
                self.params = {'method': 'minmax', 'min': float(x_min), 'max': float(x_max)}

        if self.method != 'log' and median is not None:
            df_out[col] = df_out[col].where(~df[col].isna(), np.nan)

        logging.info(f"После нормализации: min={df_out[col].min():.4f}, max={df_out[col].max():.4f}, mean={df_out[col].mean():.4f}")

        if self.method in ['minmax', 'user_minmax']:
            if df_out[col].notna().any() and (df_out[col].min() < -1e-10 or df_out[col].max() > 1 + 1e-10):
                logging.warning("Значения вне [0, 1] после MinMax-нормализации.")

        self._save_params()

        return df_out

    def denormalize(self, df, col, group='uuid', params_file=None):
        df_out = df.copy()
        if params_file is None:
            params_file = self.params_file

        try:
            with open(params_file, 'r') as f:
                params = json.load(f)
        except Exception as e:
            raise ValueError(f"Ошибка загрузки параметров: {str(e)}")

        if params['method'] != self.method:
            logging.warning(f"Метод в params ({params['method']}) не совпадает с текущим ({self.method}).")

        logging.info(f"Денормализация: метод={params['method']}")

        if params['method'] in ['minmax', 'user_minmax']:
            logging.info(f"До ограничения: min={df_out[col].min():.4f}, max={df_out[col].max():.4f}")
            df_out[col] = df_out[col].clip(lower=0, upper=1)  # [0, 1]
            logging.info(f"После ограничения: min={df_out[col].min():.4f}, max={df_out[col].max():.4f}")

        if params['method'] == 'log':
            df_out[col] = np.exp(df_out[col]) - 1
        elif params['method'] == 'user_minmax':
            for g in df_out[group].unique():
                mask = df_out[group] == g
                g_str = str(g)
                if g_str not in params['groups']:
                    logging.warning(f"Нет параметров для группы {g}. Пропускаю.")
                    continue
                x_min, x_max = params['groups'][g_str]['min'], params['groups'][g_str]['max']
                if x_max == x_min:
                    df_out.loc[mask, col] = x_min
                else:
                    df_out.loc[mask, col] = df_out.loc[mask, col] * (x_max - x_min) + x_min
        else:
            x_min, x_max = params['min'], params['max']
            if x_max == x_min:
                df_out[col] = x_min
            else:
                df_out[col] = df_out[col] * (x_max - x_min) + x_min

        logging.info(f"После денормализации: min={df_out[col].min():.2f}, max={df_out[col].max():.4f}, mean={df_out[col].mean():.2f}")

        if (df_out[col] < 0).any():
            logging.warning("Обнаружены отрицательные значения после денормализации (не должно быть так в теории). Применяю clip(0)")
            df_out[col] = df_out[col].clip(lower=0)

        return df_out