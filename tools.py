import logging
import os

def get_logger(name, log_level=logging.DEBUG, file_name="log.txt"):
    # 日志配置
    logger = logging.getLogger(name)
    logger.setLevel(level=log_level)
    handler = logging.FileHandler(file_name)
    handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(log_level)

    logger.addHandler(handler)
    logger.addHandler(console)

    # 打印测试
    logger.info("==========INFO===========")
    logger.debug("==========DEBUG===========")
    logger.warning("==========WARNING===========")
    return logger