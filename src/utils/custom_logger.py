import logging
import sys

formatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S"
)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    return logger