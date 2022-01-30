import logging


def make_logger(log_file_path: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
