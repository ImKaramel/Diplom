import logging
import yaml
import os

def load_config(config_path='config/config.yaml'):
    """
    Загружает конфигурацию из YAML-файла.

    Parameters:
    - config_path (str): Путь к конфигурационному файлу (по умолчанию 'config/config.yaml').

    Returns:
    - dict: Словарь с конфигурацией.

    Raises:
    - FileNotFoundError: Если конфигурационный файл не найден.
    - yaml.YAMLError: Если файл содержит некорректный YAML.
    """
    # Resolve path relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    config_path = os.path.join(project_root, config_path)
    print(f"Проверяемый путь к config.yaml: {config_path}")  # Отладочный вывод

    try:
        if not os.path.exists(config_path):
            logging.error(f"Конфигурационный файл не найден: {config_path}")
            raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if config is None:
            logging.error(f"Конфигурационный файл пуст: {config_path}")
            raise ValueError(f"Конфигурационный файл пуст: {config_path}")

        logging.info(f"Конфигурация успешно загружена из {config_path}")
        return config

    except yaml.YAMLError as e:
        logging.error(f"Ошибка парсинга YAML в {config_path}: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Ошибка загрузки конфигурации из {config_path}: {str(e)}")
        raise