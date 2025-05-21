import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import STL

class TimeSeriesAnalyzer:
    def __init__(self, target_column='A_plus', period=24, graphics_dir='graphics'):
        self.target_column = target_column
        self.period = period
        self.graphics_dir = graphics_dir
        self.logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
        os.makedirs(self.graphics_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    def _setup_group_logger(self, group):
        logger = logging.getLogger(f"uuid_{group}")
        logger.setLevel(logging.INFO)
        logger.handlers = []
        log_file = os.path.join(self.logs_dir, f"uuid_{group}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def statistical_summary(self, df, groupby='uuid'):
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

    def plot_time_series(self, df, groupby='uuid'):
        plt.figure(figsize=(15, 7))
        for group in df[groupby].unique()[:5]:
            group_data = df[df[groupby] == group]
            plt.plot(group_data['time_dt'], group_data[self.target_column], label=f'uuid_{group}')
        plt.xlabel('Time')
        plt.ylabel(self.target_column)
        plt.title('Time Series by UUID')
        plt.legend()
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gcf().autofmt_xdate()
        plt.savefig(os.path.join(self.graphics_dir, 'time_series_plot.png'))
        plt.close()
        logging.info("График временных рядов сохранён в time_series_plot.png")

    def plot_distribution(self, df):
        plt.figure(figsize=(10, 6))
        df[self.target_column].hist(bins=50)
        plt.title(f'Distribution of {self.target_column}')
        plt.xlabel(self.target_column)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(self.graphics_dir, 'distribution_plot.png'))
        plt.close()
        logging.info("distribution_plot.png")

    def plot_stl_decomposition(self, df, groupby='uuid'):
        """
        STL-разложение временного ряда
        """
        for group in df[groupby].unique()[:5]:
            group_logger = self._setup_group_logger(group)
            group_data = df[df[groupby] == group][self.target_column].dropna()
            if len(group_data) > self.period:
                try:
                    stl = STL(group_data, period=self.period, robust=True)
                    result = stl.fit()
                    group_logger.info(f"STL-разложение выполнено для uuid_{group}")
                    group_logger.info(f"Среднее тренда: {result.trend.mean()}")
                    group_logger.info(f"Среднее сезонности: {result.seasonal.mean()}")
                    group_logger.info(f"Среднее остатков: {result.resid.mean()}")

                    plt.figure(figsize=(10, 8))
                    plt.subplot(4, 1, 1)
                    plt.plot(group_data, label='Original')
                    plt.title(f'STL Decomposition for uuid_{group}')
                    plt.legend()

                    plt.subplot(4, 1, 2)
                    plt.plot(result.trend, label='Trend (T_t)')
                    plt.legend()

                    plt.subplot(4, 1, 3)
                    plt.plot(result.seasonal, label='Seasonal (S_t)')
                    plt.legend()

                    plt.subplot(4, 1, 4)
                    plt.plot(result.resid, label='Residuals (R_t)')
                    plt.legend()

                    plt.tight_layout()
                    plt.savefig(os.path.join(self.graphics_dir, f'stl_decomposition_{group}.png'))
                    plt.close()
                    group_logger.info(f"STL-график сохранён в stl_decomposition_{group}.png")
                except Exception as e:
                    group_logger.error(f"Ошибка STL для uuid_{group}: {str(e)}")

    def analyze(self, df, groupby='uuid'):
        logging.info("Начало анализа временных рядов")

        if df[self.target_column].isna().sum() > 0:
            logging.warning(f"Обнаружены пропуски в {self.target_column}: {df[self.target_column].isna().sum()}")

        logging.info("Статистический анализ:")
        self.statistical_summary(df, groupby)
        self.plot_time_series(df, groupby)
        self.plot_distribution(df)
        self.plot_stl_decomposition(df, groupby)

        logging.info("Анализ временных рядов завершён")