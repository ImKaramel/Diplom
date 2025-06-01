import logging
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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
            logging.warning(f"Фолд {k}: Пропущен из-за нехватки данных (обучение={len(train_df)}, проверка={len(val_df)})")

    return splits

def evaluate_forecast(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return mae, rmse

def _prepare_and_forecast(df, group, time_col, target, lookback, horizon, model, model_name, norm, plot_dir='graphics', is_normalized=False):
    logging.info(f"Подготовка данных для группы {group}, модель {model_name}")
    group_df = df[df['group'] == group].copy().sort_values(time_col)

    if group_df[time_col].duplicated().any():
        group_df = group_df.groupby(time_col).agg({target: 'mean'}).reset_index()
        group_df['group'] = group

    if len(group_df) < lookback + horizon:
        logging.warning(f"Мало данных для группы {group}, пропускаю")
        return None, None, None, None

    train_size = len(group_df) - horizon
    if train_size < lookback:
        logging.warning(f"Недостаточно данных для обучения группы {group}")
        return None, None, None, None

    logging.info(f"Разделение данных: {train_size} для обучения, {horizon} для теста")
    train_df = group_df[[time_col, target, 'group']].iloc[:train_size].copy()
    test_df = group_df[[time_col, target, 'group']].iloc[train_size:]

    logging.info(f"Обучаю {model_name.upper()} для группы {group}")
    model.train(train_df)
    logging.info(f"Создание прогноза для группы {group}")
    forecast, conf_int = model.forecast(horizon)
    forecast_vals = forecast
    test_vals = test_df[target].values
    lower_ci = conf_int['lower'].to_numpy() if conf_int is not None else np.full(horizon, np.nan)
    upper_ci = conf_int['upper'].to_numpy() if conf_int is not None else np.full(horizon, np.nan)

    if len(forecast_vals) < horizon:
        logging.warning(f"Прогноз для группы {group} короче ожидаемого ({len(forecast_vals)} вместо {horizon}), дополняю нули")
        forecast_vals = np.pad(forecast_vals, (0, horizon - len(forecast_vals)), mode='constant', constant_values=0)
    elif len(forecast_vals) > horizon:
        logging.warning(f"Прогноз для группы {group} длиннее ожидаемого ({len(forecast_vals)} вместо {horizon}), укорачиваю")
        forecast_vals = forecast_vals[:horizon]

    forecast_time = pd.date_range(
        start=group_df[time_col].iloc[-1] + pd.Timedelta(hours=1),
        periods=horizon, freq='h'
    )

    forecast_df = pd.DataFrame({
        time_col: forecast_time,
        'group': [group] * horizon,
        'forecast': forecast_vals.flatten()
    })
    if model_name != 'nbeats' and lower_ci is not None and upper_ci is not None:
        forecast_df['lower_ci'] = lower_ci.flatten()
        forecast_df['upper_ci'] = upper_ci.flatten()


    if is_normalized:
        logging.info(f"Денормализация прогноза для группы {group}")
        forecast_denorm = norm.denormalize(forecast_df, col='forecast', group='group')
        if model_name != 'nbeats' and 'lower_ci' in forecast_denorm.columns:
            forecast_denorm = norm.denormalize(forecast_denorm, col='lower_ci', group='group')
            forecast_denorm = norm.denormalize(forecast_denorm, col='upper_ci', group='group')
    else:
        forecast_denorm = forecast_df.copy()

    test_df = pd.DataFrame({
        time_col: forecast_time[:len(test_vals)],
        'target': test_vals,
        'group': [group] * len(test_vals)
    })
    if is_normalized:
        logging.info(f"Денормализация тестовых данных для группы {group}")
        test_denorm = norm.denormalize(test_df, col='target', group='group')
        test_vals_denorm = test_denorm['target'].values
    else:
        test_denorm = test_df.copy()
        test_vals_denorm = test_vals

    min_len = min(len(test_vals_denorm), len(forecast_denorm['forecast'].values))
    mae, rmse = None, None
    if min_len > 0:
        mae, rmse = evaluate_forecast(test_vals_denorm[:min_len], forecast_denorm['forecast'].values[:min_len])
        logging.info(f"Группа {group}: MAE = {mae:.4f}, RMSE = {rmse:.4f}")
    else:
        logging.warning(f"Данные для группы {group} не совпадают по длине")

    logging.info(f"Создание графика для группы {group}")
    plt.figure(figsize=(12, 6))
    plt.plot(test_denorm[time_col], test_denorm['target'], label='Реальные значения', color='blue')
    plt.plot(forecast_denorm[time_col], forecast_denorm['forecast'], label='Прогноз', color='orange')
    if model_name != 'nbeats' and 'lower_ci' in forecast_denorm.columns:
        plt.fill_between(
            forecast_denorm[time_col],
            forecast_denorm['lower_ci'],
            forecast_denorm['upper_ci'],
            color='orange', alpha=0.2, label='95% доверительный интервал'
        )
    plt.title(f"Прогноз для группы {group} ({model_name.upper()})")
    plt.xlabel('Время')
    plt.ylabel('Потребление энергии (A_plus)')
    plt.legend()
    plt.grid(True)
    os.makedirs(f"graphics/{model_name}", exist_ok=True)
    plot_path = os.path.join(f"graphics/{model_name}", f"forecast_{model_name}_{group}.png")
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"График сохранен: {plot_path}")

    return forecast_denorm, mae, rmse

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

    analyzer = TimeSeriesAnalyzer(target_column='target')
    norm_method = config.get('preprocessing', {}).get('norm_method', 'minmax')
    norm_file = config.get('data', {}).get('norm_params', 'data/processed/norm_params.json')
    norm = NormFix(method=norm_method, params_file=norm_file)
    is_normalized = config.get('preprocessing', {}).get('process_normalization', False)

    horizon = config['forecasting'].get('horizon', 24)
    lookback = config['preprocessing'].get('period', 24) * 7
    horizons = config.get('forecasting', {}).get('horizons', [72, 96, 120, 144, 168])
    max_horizon = max(horizons)

    results = []
    forecasts_df = pd.DataFrame()
    skipped_groups = []
    metrics_list = []

    for group in df['group'].unique():
        logging.info(f"Обработка группы {group}")
        group_df = df[df['group'] == group][['time_dt', 'target']].copy().set_index('time_dt')
        group_df.index = pd.to_datetime(group_df.index)

        if len(group_df) < 48:
            logging.warning(f"Мало данных для группы {group}: {len(group_df)} точек, пропускаю")
            skipped_groups.append(group)
            continue

        if not np.isfinite(group_df['target']).all():
            logging.warning(f"Некорректные данные в группе {group}, пропускаю")
            skipped_groups.append(group)
            continue

        target_series = group_df['target'].resample('h').mean().interpolate(method='linear')
        group_df = pd.DataFrame({'target': target_series, 'group': group}).reset_index().rename(columns={'index': 'time_dt'})
        logging.info(f"Группа {group}: Значения от {group_df['target'].min():.2f} до {group_df['target'].max():.2f}")

        train_df = group_df.iloc[:int(len(group_df) * 0.8)].copy()
        test_df = group_df.iloc[int(len(group_df) * 0.8):].copy()
        logging.info(f"Группа {group}: Всего {len(group_df)} точек, обучение: {len(train_df)}, тест: {len(test_df)}")

        splits = cross_validation(train_df, group_col='group', target_col='target', folds=5, horizon=max_horizon)
        if not splits:
            logging.warning(f"Проверка для группы {group} не удалась из-за нехватки данных")
            continue

        group_results = {}
        for horizon in horizons:
            mae, rmse, r2 = 0, 0, 0
            fold_count = len(splits)
            for train_fold, val_fold in splits:
                if 'time_dt' not in train_fold.columns and pd.api.types.is_datetime64_any_dtype(train_fold.index):
                    train_fold = train_fold.reset_index()
                if 'time_dt' not in val_fold.columns and pd.api.types.is_datetime64_any_dtype(val_fold.index):
                    val_fold = val_fold.reset_index()

                model = model_init(config)
                if model_name == 'arima':
                    stat_series, d = analyzer.make_stationary(
                        train_df.set_index('time_dt')['target'], max_diff=2, name=f"Группа {group}"
                    )
                    model.order = (1, d, 1)

                logging.info(f"Обучение модели для группы {group}, горизонт {horizon}, фолд")
                model.train(train_fold)
                logging.info(f"Прогноз для группы {group}, горизонт {horizon}, фолд")
                forecast, conf_int = model.forecast(horizon)
                actual = val_fold['target'].values[:horizon]
                time = val_fold['time_dt'].values[:horizon]

                min_len = min(len(time), len(forecast), len(actual), horizon)
                if min_len < horizon:
                    logging.warning(f"Укорачиваю данные для группы {group} до {min_len} точек")
                    time = time[:min_len]
                    forecast = forecast[:min_len]
                    actual = actual[:min_len]

                forecast_df = pd.DataFrame({'time_dt': time, 'target': forecast, 'group': [group] * min_len})
                if is_normalized:
                    forecast_denorm = norm.denormalize(forecast_df, col='target', group='group')['target'].values
                else:
                    forecast_denorm = forecast
                actual_df = pd.DataFrame({'time_dt': time, 'target': actual, 'group': [group] * min_len})
                if is_normalized:
                    actual_denorm = norm.denormalize(actual_df, col='target', group='group')['target'].values
                else:
                    actual_denorm = actual

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

        results.append({'group': group, 'results': group_results})

        model = model_init(config)
        if model_name == 'arima':
            stat_series, d = analyzer.make_stationary(
                train_df.set_index('time_dt')['target'], max_diff=2, name=f"Группа {group}"
            )
            model.order = (1, d, 1)

        logging.info(f"Финальный прогноз для группы {group}")
        forecast_df, mae, rmse = _prepare_and_forecast(
            df, group, 'time_dt', 'target', lookback, max_horizon, model, model_name, norm, plot_dir=analyzer.graphics_dir, is_normalized=is_normalized
        )
        if forecast_df is not None:
            forecasts_df = pd.concat([forecasts_df, forecast_df], ignore_index=True)
            logging.info(f"Прогноз для группы {group} добавлен в результаты")
            if mae is not None and rmse is not None:
                metrics_list.append({
                    'Group': group,
                    'Horizon': max_horizon,
                    'MAE': mae,
                    'RMSE': rmse,
                    'Evaluation_Type': 'Final_Forecast'
                })

    if skipped_groups:
        logging.warning(f"Пропущены группы: {', '.join(skipped_groups)}")

    if not forecasts_df.empty:
        forecast_dir = os.path.dirname(config['data']['forecasts'])
        os.makedirs(forecast_dir, exist_ok=True)
        output_path = config['data']['forecasts'].replace('.csv', f'_{model_name}.csv') if model_name == 'xgboost' else config['data']['forecasts']
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