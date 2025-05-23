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
        plt.title(f'Distribution of {self.target_column}')
        plt.xlabel(self.target_column)
        plt.ylabel('Frequency')
        plt.grid(True)
        subdir = os.path.join(self.graphics_dir, 'main_information')
        os.makedirs(subdir, exist_ok=True)
        plt.savefig(os.path.join(subdir, 'диаграмма_распределения.png'))
        plt.close()
        logging.info("Диаграмма распределения сохранена в main_information/диаграмма_распределения.png")

    def plot_stl_decomposition(self, df, groupby='group'):
        for group in df[groupby].unique()[:5]:
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


    def plot_forecast(self, df_original, df_forecast, groupby='group'):
        if self.norm_fix is None and self.config and self.config.get('preprocessing', {}).get('process_normalization', False):
            raise ValueError("Объект norm_fix не инициализирован")

        for group in df_original[groupby].unique():
            group_logger = self._setup_group_logger(group)
            original_data = df_original[df_original[groupby] == group].copy()
            forecast_data = df_forecast[df_forecast['group'] == group].copy()

            if original_data.empty or forecast_data.empty:
                group_logger.warning(f"Нет данных для визуализации прогнозов для uuid_{group}")
                continue


            if self.config and self.config.get('preprocessing', {}).get('process_normalization', False):
                original_denorm = self.norm_fix.denormalize(
                    original_data[['time_dt', self.target_column, 'group']],
                    col=self.target_column,
                    group='group'
                )
                original_denorm['time_dt'] = pd.to_datetime(original_denorm['time_dt'])

                forecast_denorm = self.norm_fix.denormalize(
                    forecast_data[['time_dt', 'A_plus_forecast', 'group']],
                    col='A_plus_forecast',
                    group='group'
                )

                if 'lower_ci' in forecast_data.columns and 'upper_ci' in forecast_data.columns:
                    forecast_denorm = self.norm_fix.denormalize(
                        forecast_denorm,
                        col='lower_ci',
                        group='group'
                    )
                    forecast_denorm = self.norm_fix.denormalize(
                        forecast_denorm,
                        col='upper_ci',
                        group='group'
                    )
            else:
                original_denorm = original_data.copy()
                original_denorm['time_dt'] = pd.to_datetime(original_denorm['time_dt'])
                forecast_denorm = forecast_data.copy()
                forecast_denorm['time_dt'] = pd.to_datetime(forecast_denorm['time_dt'])
                forecast_denorm = forecast_denorm.rename(columns={'A_plus_forecast': self.target_column})

            plt.figure(figsize=(15, 7))
            plt.plot(original_denorm['time_dt'], original_denorm[self.target_column],
                     label='Actual', color='#1f77b4', linewidth=2)
            plt.plot(forecast_denorm['time_dt'], forecast_denorm[self.target_column],
                     label='Forecast', color='#ff7f0e', linestyle='--', linewidth=2)

            if 'lower_ci' in forecast_denorm.columns and 'upper_ci' in forecast_denorm.columns:
                plt.fill_between(forecast_denorm['time_dt'],
                                 forecast_denorm['lower_ci'],
                                 forecast_denorm['upper_ci'],
                                 color='#ff9896', alpha=0.3, label='Confidence Interval')

            plt.xlabel('Time', fontsize=12)
            plt.ylabel(self.target_column, fontsize=12)
            plt.title(f'Forecast vs Actual for uuid_{group}', fontsize=14, pad=10)
            plt.legend(fontsize=10, loc='best')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
            plt.gcf().autofmt_xdate()
            plt.ylim(bottom=0)

            output_path = os.path.join(self.graphics_dir, f'forecast_plot_{group}.png')
            os.makedirs(self.graphics_dir, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            group_logger.info(f"График прогнозов сохранён в {output_path}")



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
