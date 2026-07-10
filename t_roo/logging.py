"""
Module to set up and handling logging 
"""
import logging
import sys

logger = logging.getLogger("t-roo")

def setup_logger(log_level="INFO"):
    
    try:
       level = getattr(logging, log_level.upper())
    except:
        raise ValueError(f"log_level {log_level} not understood. Must either bei 'debug', 'info', or 'warning'.")

    logger = logging.getLogger("t-roo")
    logger.setLevel(level)

    if not any([isinstance(h, logging.StreamHandler) for h in logger.handlers]):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(name)s %(levelname)-8s: %(message)s', datefmt='%H:%M'))
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)


    for handler in logger.handlers:
        handler.setLevel(level)

setup_logger()
