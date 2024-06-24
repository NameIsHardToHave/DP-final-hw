import logging

from yacs.config import CfgNode

def update_logger(logger: logging.Logger, config: CfgNode) -> None:
    r"""根据配置文件更新日志记录器"""
    # 删除已有的处理程序
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建日志处理程序，并设置日志消息的输出格式，再将日志处理程序添加到日志记录器
    if config.logger.out_file:
        file_handler = logging.FileHandler(config.logger.path+config.logger.name, mode='w')  # 输出到文件
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    if config.logger.out_console:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_formatter = logging.Formatter('%(message)s')
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

def create_logger(config: CfgNode) -> logging.Logger:
    r"""创建日志记录器并初始化"""
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    update_logger(logger, config)
    return logger