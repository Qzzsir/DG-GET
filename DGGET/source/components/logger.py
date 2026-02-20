import logging
from pathlib import Path
from typing import Tuple
from omegaconf import DictConfig


def get_formatter() -> logging.Formatter:
    """
    创建一个日志格式化器，用于定义日志消息的格式。

    输入：
    - 无

    输出：
    - logging.Formatter 对象：定义了日志消息的格式。
    """
    return logging.Formatter('[%(asctime)s][%(filename)s][L%(lineno)d][%(levelname)s] %(message)s')


def initialize_logger() -> logging.Logger:
    """
    初始化日志记录器，设置日志级别，并清除所有现有的处理器。

    输入：
    - 无

    输出：
    - logging.Logger 对象：配置了基本的日志处理器（控制台处理器）。
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()  # 关闭所有现有的处理器
    logger.handlers.clear()  # 清除所有现有的处理器

    formatter = get_formatter()  # 获取格式化器
    stream_handler = logging.StreamHandler()  # 创建流处理器（控制台输出）
    stream_handler.setFormatter(formatter)  # 设置格式化器
    logger.addHandler(stream_handler)  # 添加处理器到日志记录器

    return logger


def set_file_handler(log_file_path: Path) -> logging.Logger:
    """
    为日志记录器添加一个文件处理器，将日志消息写入指定的文件。

    输入：
    - log_file_path (Path): 要将日志写入的文件路径。

    输出：
    - logging.Logger 对象：配置了文件处理器和格式化器。
    """
    logger = initialize_logger()  # 初始化日志记录器
    formatter = get_formatter()  # 获取格式化器
    file_handler = logging.FileHandler(str(log_file_path))  # 创建文件处理器
    file_handler.setFormatter(formatter)  # 设置格式化器
    logger.addHandler(file_handler)  # 添加文件处理器到日志记录器

    return logger


def logger_factory(config: DictConfig) -> Tuple[logging.Logger]:
    """
    根据配置文件创建和返回一个配置好的日志记录器。

    输入：
    - config (DictConfig): 配置对象，包含日志文件路径 (log_path) 和唯一标识符 (unique_id)。

    输出：
    - Tuple[logging.Logger]：包含一个配置好的 logging.Logger 对象的元组。
    """
    log_path = Path(config.log_path) / config.unique_id  # 创建日志目录路径
    log_path.mkdir(exist_ok=True, parents=True)  # 创建日志目录（如果不存在）
    logger = set_file_handler(log_file_path=log_path / config.unique_id)  # 配置文件处理器并返回日志记录器
    return logger
