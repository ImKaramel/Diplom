import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.preprocessing.missing_value_handler import MissingValueHandler
from src.preprocessing.anomaly_fix import AnomalyFix
from src.preprocessing.norm_fix import NormFix

class DataPrep:
    def __init__(
            self,
            miss_method='linear',
            anom_method='stl',
            anom_act='interpolate',
            norm_method='minmax',
            graph_dir='graphics',
            process_missing=True,
            process_anomalies=True,
            process_normalization=True
    ):
        """
        - miss_method: метод заполнения пропусков ('linear', 'seasonal', 'knn')
        - anom_method: метод обнаружения аномалий ('stl', 'rf')
        - anom_act: действие с аномалиями ('interpolate', 'remove')
        - norm_method: метод нормализации ('minmax', 'log', 'user_minmax')
        - graph_dir: директория для сохранения графиков
        - process_missing: выполнять обработку пропусков (True/False)
        - process_anomalies: выполнять обработку аномалий (True/False)
        - process_normalization: выполнять нормализацию (True/False)
        """
        valid_miss = ['linear', 'seasonal', 'knn']
        valid_anom = ['stl', 'rf']
        valid_act = ['interpolate', 'remove']
        valid_norm = ['minmax', 'log', 'user_minmax']
        if miss_method not in valid_miss:
            raise ValueError(f"miss_method: {valid_miss}")
        if anom_method not in valid_anom:
            raise ValueError(f"anom_method: {valid_anom}")
        if anom_act not in valid_act:
            raise ValueError(f"anom_act: {valid_act}")
        if norm_method not in valid_norm:
            raise ValueError(f"norm_method: {valid_norm}")

        self.miss_fix = MissingValueHandler(method=miss_method) if process_missing else None
        self.anom_fix = AnomalyFix(method=anom_method, action=anom_act) if process_anomalies else None
        self.norm_fix = NormFix(method=norm_method) if process_normalization else None

        self.process_missing = process_missing
        self.process_anomalies = process_anomalies
        self.process_normalization = process_normalization

        self.miss_method = miss_method
        self.anom_method = anom_method
        self.norm_method = norm_method
        self.graph_dir = graph_dir
        os.makedirs(self.graph_dir, exist_ok=True)
        self.stats = []
        self.used_methods = []

    def _check(self, df, col, group, time_col='time_dt'):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df должен быть pandas.DataFrame")
        req_cols = [col, group, time_col]
        miss_cols = [c for c in req_cols if c not in df.columns]
        if miss_cols:
            raise ValueError(f"Нет столбцов: {miss_cols}")
        if df[time_col].isna().any():
            raise ValueError(f"Пропуски в {time_col}")
        if np.isinf(df[col]).any():
            raise ValueError(f"Обнаружены бесконечные значения в {col}")

    def _compute_stats(self, df, col, stage):
        try:
            stats = {
                'stage': stage,
                'mean': df[col].mean() if df[col].notna().any() else np.nan,
                'std': df[col].std() if df[col].notna().any() else np.nan,
                'min': df[col].min() if df[col].notna().any() else np.nan,
                'max': df[col].max() if df[col].notna().any() else np.nan,
                'missing_ratio': df[col].isna().mean(),
                'q25': df[col].quantile(0.25) if df[col].notna().any() else np.nan,
                'q50': df[col].quantile(0.50) if df[col].notna().any() else np.nan,
                'q75': df[col].quantile(0.75) if df[col].notna().any() else np.nan,
                'autocorr_lag24': df[col].autocorr(lag=24) if df[col].notna().sum() > 24 else np.nan
            }
            self.stats.append(stats)
            return stats
        except Exception as e:
            logging.error(f"Ошибка статистики на этапе {stage}: {str(e)}")
            return None

    def _plot_distribution(self, df, col, stage):
        try:
            data = df[col].dropna()
            if len(data) < 10:
                logging.warning(f"Недостаточно данных  на этапе {stage}")
                return
            plt.figure(figsize=(10, 6))
            data.hist(bins=50, density=True)
            plt.title(f'{col} ({stage})')
            plt.xlabel(col)
            plt.ylabel('Плотность')
            plt.grid(True)
            plt.savefig(os.path.join(self.graph_dir, f'dist_{stage.replace(" ", "_").lower()}.png'))
            plt.close()
        except Exception as e:
            logging.error(f"Ошибка гистограммы на этапе {stage}: {str(e)}")

    def _plot_time_series(self, dfs, labels, col, group, uuid):
        try:
            plt.figure(figsize=(15, 7))
            for df, lab in zip(dfs, labels):
                grp_data = df[df[group] == uuid]
                if grp_data[col].notna().any():
                    plt.plot(grp_data['time_dt'], grp_data[col], label=lab, alpha=0.7)
            plt.xlabel('Время')
            plt.ylabel(col)
            plt.title(f'Временной ряд для {uuid}')
            plt.legend()
            plt.grid(True)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.gcf().autofmt_xdate()
            plt.savefig(os.path.join(self.graph_dir, f'time_series_{uuid}.png'))
            plt.close()
        except Exception as e:
            logging.error(f"Ошибка временного ряда для {uuid}: {str(e)}")

    def _plot_distribution_comparison(self, dfs, labels, col):
        try:
            plt.figure(figsize=(10, 6))
            for df, lab in zip(dfs, labels):
                data = df[col].dropna()
                if len(data) < 10:
                    logging.warning(f"Недостаточно данных для сравнения на этапе {lab}")
                    continue
                data.hist(bins=50, alpha=0.5, label=lab, density=True)
            plt.title(f'Сравнение распределений {col}')
            plt.xlabel(col)
            plt.ylabel('Плотность')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.graph_dir, 'dist_comparison.png'))
            plt.close()
        except Exception as e:
            logging.error(f"Ошибка сравнения распределений: {str(e)}")

    def prepare(self, df, target, group='uuid'):
        logging.info("Начало подготовки данных")
        self._check(df, target, group)
        self.used_methods = []

        df_out = df.copy()
        dfs = [df_out]
        stages = ['исходные']

        self._compute_stats(df_out, target, 'исходные')
        self._plot_distribution(df_out, target, 'исходные')

        try:
            if self.process_missing and self.miss_fix:
                logging.info("Обработка пропусков")
                df_out = self.miss_fix.handle(df_out, target, groupby=group)
                self.used_methods.append(self.miss_method)
                if df_out[target].isna().any():
                    logging.warning(f"Остались пропуски: {df_out[target].isna().sum()}")
                dfs.append(df_out.copy())
                stages.append('после пропусков')
                self._compute_stats(df_out, target, 'после пропусков')
                self._plot_distribution(df_out, target, 'после пропусков')

            if self.process_anomalies and self.anom_fix:
                logging.info("Обработка аномалий")
                df_out = self.anom_fix.fix(df_out, target, groupby=group)
                self.used_methods.append(self.anom_method)
                dfs.append(df_out.copy())
                stages.append('после аномалий')
                self._compute_stats(df_out, target, 'после аномалий')
                self._plot_distribution(df_out, target, 'после аномалий')

            if self.process_normalization and self.norm_fix:
                logging.info("Нормализация данных")
                df_out = self.norm_fix.fix(df_out, target, group=group)
                self.used_methods.append(self.norm_method)
                dfs.append(df_out.copy())
                stages.append('после нормализации')
                self._compute_stats(df_out, target, 'после нормализации')
                self._plot_distribution(df_out, target, 'после нормализации')

            if len(dfs) > 1:
                uuid_counts = df_out.groupby(group)[target].count()
                top_uuid = uuid_counts.idxmax() if not uuid_counts.empty else df_out[group].unique()[0]
                self._plot_time_series(dfs, stages, target, group, top_uuid)
                self._plot_distribution_comparison(dfs, stages, target)

            stats_df = pd.DataFrame(self.stats)
            if not stats_df.empty:
                stats_df = stats_df.set_index('stage')
                logging.info("Статистика этапов")
                logging.info(f"\n{stats_df.to_string()}")
                print("Статистика этапов")
                try:
                    from IPython.display import display
                    display(stats_df)
                except ImportError:
                    print(stats_df)
            else:
                logging.warning("Пусто")

            logging.info(f"Подготовка данных завершена - Использованные методы: {self.used_methods}")
            return df_out, self.used_methods

        except Exception as e:
            logging.error(f"Ошибка подготовки данных: {str(e)}")
            raise