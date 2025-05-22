import logging
import os


def setup_logging(log_file='logs/project.log'):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'w'):
        pass
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )


def setup_group_logging(group, logs_dir='logs'):
    logger = logging.getLogger(f"uuid_{group}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    log_file = os.path.join(logs_dir, f"uuid_{group}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
