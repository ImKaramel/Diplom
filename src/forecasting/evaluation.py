import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.forecasting.arima_model import ARIMAModel
from src.preprocessing import NormFix


def time_series_cross_validation(series, n_splits=2, horizon=24):
    n_train = len(series)
    if n_train < horizon * 2:
        return []

    min_train_length = max(48, horizon * 2)
    step_size = (n_train - horizon) // n_splits

    if step_size < min_train_length:
        step_size = min_train_length

    splits = []
    for i in range(n_splits):
        train_start = i * step_size
        train_end = train_start + step_size
        val_end = train_end + horizon

        if val_end > n_train:
            val_end = n_train
            train_end = max(0, val_end - horizon - step_size)
            train_start = max(0, train_end - step_size)

        train_data = series[train_start:train_end]
        val_data = series[train_end:val_end]

        if len(train_data) >= min_train_length and len(val_data) >= horizon:
            splits.append((train_data, val_data))

    return splits

def evaluate_forecast(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    return mae, rmse

def forecast_arima():
    try:
        file = 'clean_data.csv'
        logging.info(f"Читаю {file}")
        if not os.path.exists(file):
            raise FileNotFoundError(f"Файл не найден: {file}")
        data = pd.read_csv(file, sep='\t')
    except Exception as e:
        logging.error(f"Ошибка загрузки: {str(e)}")
        raise

    need_cols = ['time_dt', 'A_plus', 'uuid']
    miss_cols = [c for c in need_cols if c not in data.columns]
    if miss_cols:
        raise KeyError(f"Нет столбцов: {miss_cols}. Есть: {data.columns.tolist()}")

    data['time_dt'] = pd.to_datetime(data['time_dt'], errors='raise')
    data = data.set_index('time_dt')

    norm_fix = NormFix()
    try:
        norm_params_file = 'norm_params.json'
        if not os.path.exists(norm_params_file):
            logging.warning(f"Файл параметров нормализации {norm_params_file} не найден, денормализация невозможна")
            norm_params_file = None
    except Exception as e:
        logging.error(f"Ошибка загрузки параметров нормализации: {str(e)}")
        norm_params_file = None

    horizons = [24]
    forecasts = []
    results = {}
    skipped_groups = []

    for uuid in data['uuid'].unique():
        group_data = data[data['uuid'] == uuid]['A_plus']
        if len(group_data) < 48:
            logging.warning(f"Мало данных для uuid={uuid}: {len(group_data)} наблюдений, пропускаю")
            skipped_groups.append(uuid)
            continue

        if not np.isfinite(group_data).all():
            logging.warning(f"Данные для uuid={uuid} содержат NaN или inf, пропускаю")
            skipped_groups.append(uuid)
            continue

        group_data = group_data.resample('h').mean().interpolate(method='linear')
        group_data.index.freq = 'h'

        logging.info(f"uuid={uuid}: Диапазон A_plus: min={group_data.min():.2f}, max={group_data.max():.2f}")

        n_train = int(len(group_data) * 0.8)
        train_data = group_data[:n_train]
        test_data = group_data[n_train:]
        logging.info(f"uuid={uuid}: Общая длина: {len(group_data)}, тренировочная: {len(train_data)}, тестовая: {len(test_data)}")

        if len(train_data) < 72:
            logging.warning(f"Недостаточно данных для кросс-валидации uuid={uuid}, использую все данные")
            cv_maes = []
        else:
            logging.info(f"Кросс-валидация для uuid={uuid}")
            cv_splits = time_series_cross_validation(train_data, n_splits=2, horizon=max(horizons))
            cv_maes = []

            for train_split, val_split in cv_splits:
                logging.info(f"Фолд: train={len(train_split)}, val={len(val_split)}")
                if len(train_split) < 24 or len(val_split) < max(horizons):
                    logging.warning(f"Недостаточно данных для фолда, пропускаю")
                    continue
                try:
                    arima = ARIMAModel(auto=True, s=24)
                    arima.train(train_split)
                    forecast = arima.forecast(steps=len(val_split))
                    if len(forecast) < len(val_split):
                        raise ValueError("Недостаточная длина прогноза")
                    mae, _ = evaluate_forecast(val_split, forecast)
                    cv_maes.append(mae)
                except Exception as e:
                    logging.error(f"Ошибка кросс-валидации для uuid={uuid}: {str(e)}")
                    continue

            if cv_maes:
                logging.info(f"Среднее MAE на кросс-валидации для uuid={uuid}: {np.mean(cv_maes):.2f}")
            else:
                logging.warning(f"Кросс-валидация для uuid={uuid} не выполнена")

        logging.info(f"Обучаю модель для uuid={uuid}")
        arima = ARIMAModel(auto=True, s=24)
        try:
            arima.train(train_data)
        except Exception as e:
            logging.error(f"Ошибка обучения для uuid={uuid}: {str(e)}")
            skipped_groups.append(uuid)
            continue

        for horizon in horizons:
            forecast = arima.forecast(steps=horizon)
            # Денормализация прогноза
            forecast_df = pd.DataFrame({
                'uuid': [uuid] * len(forecast),
                'time_dt': pd.date_range(start=train_data.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq='h'),
                'forecast': forecast
            })
            if norm_params_file:
                try:
                    forecast_df = norm_fix.denormalize(forecast_df, col='forecast', group='uuid', params_file=norm_params_file)
                except Exception as e:
                    logging.error(f"Ошибка денормализации для uuid={uuid}: {str(e)}")

            actual = test_data[:horizon]
            mae, rmse = evaluate_forecast(actual, forecast)
            logging.info(f"uuid={uuid}, горизонт={horizon}: MAE={mae:.2f}, RMSE={rmse:.2f}")
            forecasts.append(forecast_df)
            results[(uuid, horizon)] = (mae, rmse)

    if skipped_groups:
        logging.info(f"Пропущено групп: {len(skipped_groups)}, uuid: {skipped_groups}")

    try:
        if forecasts:
            forecasts_df = pd.concat(forecasts, ignore_index=True)
            forecasts_df.to_csv('forecasts.csv', index=False, sep='\t')
            logging.info("Прогнозы сохранены в forecasts.csv")
        else:
            logging.warning("Нет прогнозов для сохранения")
    except Exception as e:
        logging.error(f"Ошибка сохранения прогнозов {str(e)}")
        raise

    return results, forecasts_df if forecasts else None

# if __name__ == "__main__":
#     results, forecasts_df = forecast_arima()