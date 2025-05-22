import logging
import yaml
import os


def load_config(config_path='config/config.yaml'):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    config_path = os.path.join(project_root, config_path)
    print(f"путь к config.yaml: {config_path}")

    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Конфигурационный файл не найден {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"Конфигурационный файл пуст {config_path}")

        logging.info(f"Конфигурация успешно загружена из {config_path}")
        return config

    except yaml.YAMLError as e:
        logging.error(f"Ошибка парсинга YAML в {config_path}: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Ошибка загрузки конфигурации из {config_path}: {str(e)}")
        raise
