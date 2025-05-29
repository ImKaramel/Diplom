import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from src.utils.logging_config import setup_group_logging


class TimeSeriesAnalyzer:
    def __init__(self, target_column='A_plus', period=24, graphics_dir='graphics', norm_fix=None, config=None):
        self.target_column = target_column
        self.period = period
        self.graphics_dir = graphics_dir
        self.norm_fix = norm_fix
        self.logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
        self.config = config
        os.makedirs(self.graphics_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    def _setup_group_logger(self, group):
        return setup_group_logging(group, logs_dir=self.logs_dir)

    def statistical_summary(self, df, groupby='group'):
        stats = {
            'mean': df[self.target_column].mean(),
            'std': df[self.target_column].std(),
            'min': df[self.target_column].min(),
            'max': df[self.target_column].max(),
            'missing_ratio': df[self.target_column].isna().mean()
        }
        for key, value in stats.items():
            logging.info(f"{key}: {value}")

        for group in df[groupby].unique():
            group_logger = self._setup_group_logger(group)
            group_data = df[df[groupby] == group][self.target_column]
            group_logger.info(f"uuid_{group}_mean: {group_data.mean()}")
            group_logger.info(f"uuid_{group}_std: {group_data.std()}")
            group_logger.info(f"uuid_{group}_min: {group_data.min()}")
            group_logger.info(f"uuid_{group}_max: {group_data.max()}")
            group_logger.info(f"uuid_{group}_missing_ratio: {group_data.isna().mean()}")

    def plot_time_series(self, df, groupby='group'):
        plt.figure(figsize=(15, 7))
        for group in df[groupby].unique()[:5]:
            group_data = df[df[groupby] == group].copy()
            group_data['time_dt'] = pd.to_datetime(group_data['time_dt'])
            plt.plot(group_data['time_dt'], group_data[self.target_column], label=f'uuid_{group}')
        plt.xlabel('Время')
        plt.ylabel(self.target_column)
        plt.title('Временные ряды UUID')
        plt.legend()
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gcf().autofmt_xdate()
        subdir = os.path.join(self.graphics_dir, 'main_information')
        os.makedirs(subdir, exist_ok=True)
        plt.savefig(os.path.join(subdir, 'временные_ряды.png'))
        plt.close()
        logging.info("График временных рядов сохранён в main_information/временные_ряды.png")

    def plot_distribution(self, df):
        plt.figure(figsize=(10, 6))
        df[self.target_column].hist(bins=50)
        plt.title(f'Распределение {self.target_column}')
        plt.xlabel(self.target_column)
        plt.ylabel('Частота')
        plt.grid(True)
        subdir = os.path.join(self.graphics_dir, 'main_information')
        os.makedirs(subdir, exist_ok=True)
        plt.savefig(os.path.join(subdir, 'диаграмма_распределения.png'))
        plt.close()
        logging.info("Диаграмма распределения сохранена в main_information/диаграмма_распределения.png")

    def plot_stl_decomposition(self, df, groupby='group'):
        all_trends = []
        all_seasonals = []
        all_resids = []

        for group in df[groupby].unique():
            group_logger = self._setup_group_logger(group)
            group_data = df[df[groupby] == group].copy()
            group_data['time_dt'] = pd.to_datetime(group_data['time_dt'])
            group_data = group_data.set_index('time_dt')[self.target_column].dropna()
            if len(group_data) > self.period:
                try:
                    stl = STL(group_data, period=self.period, robust=True)
                    result = stl.fit()
                    group_logger.info(f"STL-разложение выполнено для uuid_{group}")
                    group_logger.info(f"Среднее тренда: {result.trend.mean()}")
                    group_logger.info(f"Среднее сезонности: {result.seasonal.mean()}")
                    group_logger.info(f"Среднее остатков: {result.resid.mean()}")

                    all_trends.extend(result.trend.dropna())
                    all_seasonals.extend(result.seasonal.dropna())
                    all_resids.extend(result.resid.dropna())

                    plt.figure(figsize=(10, 8))
                    plt.subplot(4, 1, 1)
                    plt.plot(group_data, label='Исходный')
                    plt.title(f'Декомпозиция STL для uuid_{group}')
                    plt.legend()

                    plt.subplot(4, 1, 2)
                    plt.plot(result.trend, label='Тренд (T_t)')
                    plt.legend()

                    plt.subplot(4, 1, 3)
                    plt.plot(result.seasonal, label='Сезонность (S_t)')
                    plt.legend()

                    plt.subplot(4, 1, 4)
                    plt.plot(result.resid, label='Остатки (R_t)')
                    plt.legend()

                    plt.tight_layout()
                    subdir = os.path.join(self.graphics_dir, 'main_information')
                    os.makedirs(subdir, exist_ok=True)
                    plt.savefig(os.path.join(subdir, f'stl_decomposition_{group}.png'))
                    plt.close()
                    group_logger.info(f"STL-график сохранён в main_information/stl_decomposition_{group}.png")
                except Exception as e:
                    group_logger.error(f"Ошибка STL для uuid_{group}: {str(e)}")

        if all_trends and all_seasonals and all_resids:
            avg_trend = pd.Series(all_trends).mean()
            avg_seasonal = pd.Series(all_seasonals).mean()
            avg_resid = pd.Series(all_resids).mean()
            logging.info(f"Средний тренд по всем группам: {avg_trend}")
            logging.info(f"Средняя сезонность по всем группам: {avg_seasonal}")
            logging.info(f"Средний остаточный компонент по всем группам: {avg_resid}")

    def analyze(self, df, groupby='group'):
        logging.info("Начало анализа временных рядов")
        df = df.copy()
        df['time_dt'] = pd.to_datetime(df['time_dt'])

        if df[self.target_column].isna().sum() > 0:
            logging.warning(f"Обнаружены пропуски в {self.target_column}: {df[self.target_column].isna().sum()}")

        logging.info("Статистический анализ:")
        self.statistical_summary(df, groupby)
        self.plot_time_series(df, groupby)
        self.plot_distribution(df)
        self.plot_stl_decomposition(df, groupby)

        logging.info("Анализ временных рядов завершён")

    def test_stationarity(self, series, name="Series"):
        series = series.dropna()

        if len(series) < 24:
            logging.warning(f"Слишком мало данных для теста ADF в {name}: {len(series)} наблюдений")
            return False

        result = adfuller(series)
        p_value = result[1]
        adf_statistic = result[0]
        critical_values = result[4]

        logging.info(f"ADF тест для {name}:")
        logging.info(f"ADF Statistic: {adf_statistic:.4f}")
        logging.info(f"p-value: {p_value:.4f}")
        logging.info("Критические значения:")
        for key, value in critical_values.items():
            logging.info(f"\t{key}: {value:.4f}")

        is_stationary = p_value < 0.05
        if is_stationary:
            logging.info(f"{name} стационарен (p-value < 0.05)")
        else:
            logging.info(f"{name} не стационарен (p-value >= 0.05), требуется дифференцирование")

        output_path = os.path.join(self.graphics_dir, f"{name}_adf_test.txt")
        os.makedirs(self.graphics_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(f"ADF тест для {name}:\n")
            f.write(f"ADF Statistic: {adf_statistic:.4f}\n")
            f.write(f"p-value: {p_value:.4f}\n")
            f.write("Критические значения:\n")
            for key, value in critical_values.items():
                f.write(f"\t{key}: {value:.4f}\n")
            f.write(f"{name} {'стационарен' if is_stationary else 'не стационарен'} (p-value {'<' if p_value < 0.05 else '>='} 0.05)\n")
        logging.info(f"Результаты ADF теста сохранены в {output_path}")

        return is_stationary

    def make_stationary(self, series, max_diff=2, name="Series"):
        diff_series = series.copy()
        d = 0

        if self.test_stationarity(diff_series, name):
            return diff_series, d

        for i in range(1, max_diff + 1):
            diff_series = diff_series.diff().dropna()
            d = i
            logging.info(f"Дифференцирование порядка {d} для {name}")
            if self.test_stationarity(diff_series, f"{name} после дифференцирования {d}"):
                break

        return diff_series, d

    def plot_acf_pacf(self, series, lags=40, title="Series"):
        series = series.dropna()

        if len(series) < lags:
            logging.warning(f"Слишком мало данных для построения ACF/PACF: {len(series)} наблюдений")
            return

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plot_acf(series, lags=lags, ax=plt.gca())
        plt.title(f"ACF для {title}")

        plt.subplot(1, 2, 2)
        plot_pacf(series, lags=lags, ax=plt.gca())
        plt.title(f"PACF для {title}")

        plt.tight_layout()

        output_path = os.path.join(self.graphics_dir, f"{title}_acf_pacf.png")
        os.makedirs(self.graphics_dir, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Графики ACF и PACF сохранены в {output_path}")
