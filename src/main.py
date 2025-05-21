import pandas as pd
import logging
import os
from src.utils import setup_logging, load_config
from src.preprocessing.data_prep import DataPrep
from src.preprocessing.time_series_analyzer import TimeSeriesAnalyzer


def main():
    print(f"Текущая директория: {os.getcwd()}")
    setup_logging()
    config = load_config()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(project_root, config['data']['raw'])
    logging.info(f"Загрузка данных из {data_path}")
    data = pd.read_csv(data_path, sep='\t')
    data['time_dt'] = pd.to_datetime(data['time_dt'])

    analyzer = TimeSeriesAnalyzer(
        target_column=config['preprocessing']['target_column'],
        period=config['preprocessing']['period'],
        graphics_dir=config['graphics']['output_dir'],
        logs_dir=config['logging'].get('series_logs_dir', 'src/logs')
    )
    analyzer.analyze(data, groupby='uuid')

    prep = DataPrep(
        miss_method=config['preprocessing']['miss_method'],
        anom_method=config['preprocessing']['anom_method'],
        anom_act=config['preprocessing']['anom_act'],
        norm_method=config['preprocessing']['norm_method'],
        graph_dir=config['graphics']['output_dir'],
        process_missing=config['preprocessing'].get('process_missing', True),
        process_anomalies=config['preprocessing'].get('process_anomalies', True),
        process_normalization=config['preprocessing'].get('process_normalization', True)
    )
    clean_data, used_methods = prep.prepare(data, target='A_plus', group='uuid')

    methods_suffix = '_'.join(used_methods) if used_methods else 'raw'
    processed_dir = os.path.dirname(os.path.join(project_root, config['data']['processed']))
    processed_filename = f"clean_data_{methods_suffix}.csv"
    processed_path = os.path.join(processed_dir, processed_filename)

    os.makedirs(processed_dir, exist_ok=True)
    clean_data.to_csv(processed_path, index=False, sep='\t')

    logging.info(f"Обработанные данные сохранены в {processed_path}")


if __name__ == "__main__":
    main()
