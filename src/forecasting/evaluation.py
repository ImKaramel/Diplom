import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.forecasting.arima_model import ARIMAModel
from src.forecasting.nbeats_model import NBEATSInitializer
from src.forecasting.xgboost_model import XGBoostModel
from src.preprocessing import TimeSeriesAnalyzer
from src.preprocessing.norm_fix import NormFix

def cross_validation(df, group_col, target_col, folds=5, horizon=24):
    series = df[target_col]
    total_points = len(series)
    if total_points < horizon * 2:
        logging.warning(f"Мало данных для проверки: только {total_points} точек")
        return []

    min_train = max(48, horizon * 2)
    fold_size = (total_points - horizon) // min_train
    folds = min(folds, fold_size)
    if folds <= 0:
        logging.warning(f"Невозможно создать фолды: данных недостаточно")
        return []

    fold_size = (total_points - horizon) // folds
    splits = []
    for k in range(1, folds + 1):
        train_end = fold_size * k
        val_start = train_end
        val_end = min(val_start + horizon, total_points)

        if train_end < min_train:
            logging.warning(f"Фолд {k}: Мало данных для обучения ({train_end} точек)")
            continue

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[val_start:val_end].copy()

        if len(train_df) >= min_train and len(val_df) > 0:
            splits.append((train_df, val_df))
            logging.info(f"Фолд {k}: {len(train_df)} точек для обучения, {len(val_df)} для проверки")
        else:
            logging.warning(f"Фолд {k}: Пропущен из-за нехватки данных")

    return splits

def evaluate_forecast(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return mae, rmse

def plot_forecast(actual_df, forecast_df, group, output_dir, norm, model):
    plt.figure(figsize=(12, 6))


    actual_df_denorm = actual_df.copy()
    if 'actual' in actual_df_denorm.columns:
        actual_df_denorm = norm.denormalize(actual_df_denorm, col='actual', group=group)
        actual_values = actual_df_denorm['actual'].values
    else:
        actual_values = actual_df['actual'].values  # Если данные уже денормализованы

    plt.plot(actual_df['time_dt'], actual_values, label='Actual', color='blue')
    plt.plot(forecast_df['time_dt'], forecast_df['forecast'], label='Forecast', color='red', linestyle='--')

    plt.title(f'Forecast vs Actual for Group {group}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)


    os.makedirs(f"{output_dir}/{model}", exist_ok=True)
    plot_path = os.path.join(f"{output_dir}/{model}", f'{model}_forecast_{group}.png')
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"График сохранён в {plot_path}")

def _prepare_and_forecast(model, train_df, test_df, time_col, target, lookback, norm, final_horizon, is_normalized=False):
    logging.info(f"Подготовка прогноза для горизонта {final_horizon}")
    forecast, _ = model.forecast(final_horizon) if not hasattr(model, 'forecast_recursive') else model.forecast_recursive(final_horizon, lookback, train_df)

    forecast_time = pd.date_range(
        start=train_df[time_col].iloc[-1] + pd.Timedelta(hours=1),
        periods=final_horizon, freq='h'
    )
    forecast_df = pd.DataFrame({
        time_col: forecast_time,
        'group': train_df['group'].iloc[0],
        'forecast': forecast.flatten()
    })

    if is_normalized:
        logging.info(f"Денормализация прогноза")
        forecast_df = norm.denormalize(forecast_df, col='forecast', group='group')


    test_df = test_df.copy().rename(columns={target: 'actual'})
    if len(test_df) > final_horizon:
        test_df = test_df.iloc[-final_horizon:]
    if is_normalized:
        test_denorm = norm.denormalize(test_df, col='actual', group='group')
        actual_denorm = test_denorm['actual'].values
        forecast_denorm = forecast_df['forecast'].values[:len(actual_denorm)]  # Обрезаем прогноз до длины actual
    else:
        actual_denorm = test_df['actual'].values
        forecast_denorm = forecast_df['forecast'].values[:len(actual_denorm)]

    min_len = min(len(actual_denorm), len(forecast_denorm))
    actual_denorm = actual_denorm[:min_len]
    forecast_denorm = forecast_denorm[:min_len]

    mae, rmse = evaluate_forecast(actual_denorm, forecast_denorm) if min_len > 0 else (None, None)
    if mae is not None and rmse is not None:
        logging.info(f"MAE={mae:.2f}, RMSE={rmse:.2f}")

    return forecast_df, test_df, mae, rmse

def forecast(config, data, file_path, model_name, model_init):
    logging.info(f"Запуск прогнозирования с моделью {model_name.upper()}")

    if data is None:
        if file_path is None:
            raise ValueError("Укажите данные или путь к файлу")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл {file_path} не найден")
        df = pd.read_csv(file_path, sep='\t')
    else:
        df = data.copy()

    columns = config.get('data', {}).get('column_mapping', {'group': 'uuid', 'target': 'A_plus'})
    df = df.rename(columns={v: k for k, v in columns.items()})
    df['time_dt'] = pd.to_datetime(df['time_dt'])

    analyzer = TimeSeriesAnalyzer(target_column='target', config=config)
    norm_method = config.get('preprocessing', {}).get('norm_method', 'minmax')
    norm_file = config.get('data', {}).get('norm_params', 'data/processed/norm_params.json')
    norm = NormFix(method=norm_method, params_file=norm_file)
    is_normalized = config.get('preprocessing', {}).get('process_normalization', False)

    horizon = config['forecasting'].get('horizon', 168)
    lookback = config['preprocessing'].get('lookback', 168)
    horizons = config.get('forecasting', {}).get('horizons', [168, 360])
    final_horizon = config['forecasting'].get('final_horizon', 720)

    results = []
    forecasts_df = pd.DataFrame()
    skipped_groups = []
    metrics_list = []

    for group in df['group'].unique():
        logging.info(f"Обработка группы {group}")
        group_df = df[df['group'] == group][['time_dt', 'target', 'group']].copy().sort_values('time_dt')
        logging.info(f"Всего точек для группы {group}: {len(group_df)}")

        if len(group_df) < lookback + final_horizon:
            logging.warning(f"Мало данных для группы {group}: {len(group_df)} точек, требуется {lookback + final_horizon}, пропускаю")
            skipped_groups.append(group)
            continue

        if not np.isfinite(group_df['target']).all():
            logging.warning(f"Некорректные данные в группе {group}, пропускаю")
            skipped_groups.append(group)
            continue

        # Разделение на тренировочные и тестовые данные (последние final_horizon точек как тестовый набор)
        test_df = group_df.iloc[-final_horizon:].copy()
        train_df = group_df.iloc[:-final_horizon].copy()
        logging.info(f"Группа {group}: {len(train_df)} точек для обучения, {len(test_df)} для теста")

        if len(test_df) == 0 or len(train_df) < lookback:
            logging.warning(f"Недостаточно данных для группы {group}: нет тестового набора или мало тренировочных данных")
            skipped_groups.append(group)
            continue

        # Кросс-валидация
        splits = cross_validation(train_df, group_col='group', target_col='target', folds=2, horizon=max(horizons))
        if not splits:
            logging.warning(f"Проверка для группы {group} не удалась из-за нехватки данных")
            continue

        group_results = {}
        for horizon in horizons:
            mae, rmse = 0, 0
            fold_count = len(splits)
            for train_fold, val_fold in splits:
                model = model_init(config)
                if model_name == 'arima':
                    stat_series, d = analyzer.make_stationary(train_fold.set_index('time_dt')['target'], max_diff=2)
                    model.order = (1, d, 1)

                model.train(train_fold)
                forecast, _ = model.forecast(horizon)
                actual = val_fold['target'].values[:horizon]
                time = val_fold['time_dt'].values[:horizon]

                min_len = min(len(time), len(forecast), len(actual), horizon)
                forecast_df = pd.DataFrame({'time_dt': time[:min_len], 'target': forecast[:min_len], 'group': [group] * min_len})
                actual_df = pd.DataFrame({'time_dt': time[:min_len], 'target': actual[:min_len], 'group': [group] * min_len})

                if is_normalized:
                    forecast_denorm = norm.denormalize(forecast_df, col='target', group='group')['target'].values
                    actual_denorm = norm.denormalize(actual_df, col='target', group='group')['target'].values
                else:
                    forecast_denorm, actual_denorm = forecast[:min_len], actual[:min_len]

                fold_mae, fold_rmse = evaluate_forecast(actual_denorm, forecast_denorm)
                mae += fold_mae
                rmse += fold_rmse

            mae /= fold_count
            rmse /= fold_count
            group_results[horizon] = {'MAE': mae, 'RMSE': rmse}
            logging.info(f"Группа {group}, горизонт {horizon}: MAE={mae:.2f}, RMSE={rmse:.2f}")
            metrics_list.append({
                'Group': group,
                'Horizon': horizon,
                'MAE': mae,
                'RMSE': rmse,
                'Evaluation_Type': 'Cross-Validation'
            })

        # Финальный прогноз для последних final_horizon точек
        model = model_init(config)
        if model_name == 'arima':
            stat_series, d = analyzer.make_stationary(train_df.set_index('time_dt')['target'], max_diff=2)
            model.order = (1, d, 1)

        model.train(train_df)
        forecast_df, test_df, mae, rmse = _prepare_and_forecast(model, train_df, test_df, 'time_dt', 'target', lookback, norm, final_horizon, is_normalized)

        if forecast_df is not None:
            forecasts_df = pd.concat([forecasts_df, forecast_df], ignore_index=True)
            plot_forecast(test_df, forecast_df, group, os.path.dirname(config['data']['forecasts']), norm, model_name)
            if mae is not None and rmse is not None:
                metrics_list.append({
                    'Group': group,
                    'Horizon': final_horizon,
                    'MAE': mae,
                    'RMSE': rmse,
                    'Evaluation_Type': 'Final_Forecast'
                })

    if skipped_groups:
        logging.warning(f"Пропущены группы: {', '.join(skipped_groups)}")

    if not forecasts_df.empty:
        forecast_dir = os.path.dirname(config['data']['forecasts'])
        os.makedirs(forecast_dir, exist_ok=True)
        output_path = config['data']['forecasts'].replace('.csv', f'_{model_name}.csv')
        forecasts_df.to_csv(output_path, index=False, sep='\t')
        logging.info(f"Прогнозы сохранены в {output_path}")

        metrics_df = pd.DataFrame(metrics_list)
        metrics_path = os.path.join(forecast_dir, f'metrics_{model_name}.csv')
        metrics_df.to_csv(metrics_path, index=False, sep='\t')
        logging.info(f"Метрики сохранены в {metrics_path}")
    else:
        logging.warning("Прогнозы не созданы")

    logging.info("Прогнозирование завершено")
    return results, forecasts_df

def forecast_arima(config, data=None, file_path=None):
    return forecast(config, data, file_path, 'arima', lambda cfg: ARIMAModel(cfg))

def forecast_nbeats(config, data=None, file_path=None):
    return forecast(config, data, file_path, 'nbeats', lambda cfg: NBEATSInitializer(cfg).initialize_model())

def forecast_xgboost(config, data=None, file_path=None):
    return forecast(config, data, file_path, 'xgboost', lambda cfg: XGBoostModel(cfg))