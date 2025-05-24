import logging
import os
import pandas as pd
from utils.logging_config import setup_logging
from utils.load_config import load_config
from preprocessing.time_series_analyzer import TimeSeriesAnalyzer
from forecasting.evaluation import forecast_arima, forecast_nbeats, forecast_xgboost
from preprocessing.norm_fix import NormFix
from preprocessing.data_prep import DataPrep

def main():
    setup_logging(log_file='logs/project.log')

    config = load_config(config_path='config/config.yaml')

    processed_path = config['data']['processed']
    full_processed_path = os.path.abspath(processed_path)
    logging.info(f"путь к обработанным данным {full_processed_path}")

    norm_method = config.get('preprocessing', {}).get('norm_method', 'minmax')
    miss_method = config.get('preprocessing', {}).get('miss_method', 'linear')
    anom_method = config.get('preprocessing', {}).get('anom_method', 'stl')
    expected_filename = f"clean_data_{miss_method}_{anom_method}_{norm_method}.csv"
    expected_full_path = os.path.join(os.path.dirname(full_processed_path), expected_filename)
    logging.info(f"путь к обработанным данным (составленный)  {expected_full_path}")

    norm_fix = None
    if os.path.exists(expected_full_path):
        logging.info(f"Файл с обработанными данными найден (заново предобрабатывать не нужно): {expected_full_path}")
        clean_data = pd.read_csv(expected_full_path, sep='\t')
        clean_data['time_dt'] = pd.to_datetime(clean_data['time_dt'])
        norm_params_file = config.get('data', {}).get('norm_params', 'data/processed/norm_params.json')
        norm_fix = NormFix(method=norm_method, params_file=norm_params_file)
    else:
        logging.info(f"Файл {expected_full_path} не найден, начинаем анализ и предобработку")
        raw_data_path = config['data'].get('raw', 'data/raw/data.csv')
        if not os.path.exists(raw_data_path):
            logging.error(f"Исходный файл не найден {raw_data_path}")
            raise FileNotFoundError(f"Файл {raw_data_path} не найден")

        raw_data = pd.read_csv(raw_data_path, sep='\t')
        logging.info(f"Загружены исходные данные {raw_data_path}")

        start_date = raw_data['time_dt'].min()
        end_date = raw_data['time_dt'].max()
        num_users = raw_data['uuid'].nunique()
        min_value = raw_data['A_plus'].min()
        max_value = raw_data['A_plus'].max()
        logging.info(f"Информация об исходном датасете (до предобработки):")
        logging.info(f"Период охвата: с {start_date} по {end_date}")
        logging.info(f"Количество пользователей - {num_users}")
        logging.info(f"Минимальное значение A_plus - {min_value}")
        logging.info(f"Максимальное значение A_plus - {max_value}")

        analyzer = TimeSeriesAnalyzer(target_column='A_plus', period=config['preprocessing'].get('period', 24),
                                      graphics_dir=config['graphics'].get('output_dir', 'graphics'), config=config)
        analyzer.analyze(raw_data, groupby='uuid')
        logging.info("Анализ временных рядов завершён")

        data_prep = DataPrep(config)
        clean_data, used_methods = data_prep.prepare(raw_data, target='A_plus', group='uuid')
        logging.info(f"Использованные методы предобработки ---  {used_methods}")
        logging.info(f"Размер clean_data после предобработки -- {clean_data.shape}")
        norm_fix = data_prep.norm_fix

        clean_data['time_dt'] = pd.to_datetime(clean_data['time_dt'])
        os.makedirs(os.path.dirname(expected_full_path), exist_ok=True)
        clean_data.to_csv(expected_full_path, sep='\t', index=False)
        logging.info(f"Предобработанные данные сохранены в: {expected_full_path}")


    column_mapping = config.get('data', {}).get('column_mapping', {
        'group': 'uuid',
        'target': 'A_plus'
    })
    mapped_cols = {v: k for k, v in column_mapping.items()}
    clean_data = clean_data.rename(columns=mapped_cols)

    if norm_fix is None:
        norm_params_file = config.get('data', {}).get('norm_params', 'data/processed/norm_params.json')
        norm_fix = NormFix(method=norm_method, params_file=norm_params_file)

    analyzer = TimeSeriesAnalyzer(
        target_column='target',
        period=config['preprocessing']['period'],
        graphics_dir=config['graphics']['output_dir'],
        norm_fix=norm_fix
    )
    model_type = config['forecasting'].get('model', 'arima').lower()
    logging.info(f"Начало прогнозирования с помощью {model_type.upper()}")

    if model_type == 'arima':
        results, forecasts_df = forecast_arima(config, data=clean_data)
    elif model_type == 'nbeats':
        results, forecasts_df = forecast_nbeats(config, data=clean_data)
    elif model_type == 'xgboost':
        results, forecasts_df = forecast_xgboost(config, data=clean_data)
    else:
        raise ValueError(f"Неподдерживаемая модель: {model_type}. Используйте 'arima', 'nbeats' или 'xgboost'.")

    logging.info("Прогнозирование завершено")


if __name__ == "__main__":
    main()