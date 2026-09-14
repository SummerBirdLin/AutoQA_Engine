"""
日志与耗时追踪工具模块
"""

import logging
import sys
import time
from typing import Optional


# ANSI 颜色码
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_RED = "\033[31m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_BLUE = "\033[34m"
COLOR_CYAN = "\033[36m"
COLOR_GRAY = "\033[90m"


class ColoredFormatter(logging.Formatter):
    """带颜色的控制台日志格式化器"""

    LEVEL_COLORS = {
        logging.DEBUG: COLOR_GRAY,
        logging.INFO: COLOR_CYAN,
        logging.WARNING: COLOR_YELLOW,
        logging.ERROR: COLOR_RED,
        logging.CRITICAL: COLOR_RED + COLOR_BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, COLOR_RESET)
        time_str = self.formatTime(record, "%H:%M:%S")
        prefix = f"{COLOR_GRAY}[{time_str}]{COLOR_RESET} {color}[{record.levelname:<5}]{COLOR_RESET}"
        name_str = f"{COLOR_BOLD}{record.name}{COLOR_RESET}"
        return f"{prefix} {name_str}: {record.getMessage()}"


def get_logger(name: str = "AutoQA", level: int = logging.INFO) -> logging.Logger:
    """获取格式化日志记录器"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColoredFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger


logger = get_logger("AutoQA")


class TimerContext:
    """耗时计时器上下文管理器"""

    def __init__(self, step_name: str, log_result: bool = True, log_level: int = logging.INFO):
        self.step_name = step_name
        self.log_result = log_result
        self.log_level = log_level
        self.start_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
        if self.log_result:
            status_color = COLOR_GREEN if self.elapsed_ms < 1000 else (COLOR_YELLOW if self.elapsed_ms < 2500 else COLOR_RED)
            logger.log(
                self.log_level,
                f"⏱️  [{self.step_name}] 耗时: {status_color}{self.elapsed_ms:.2f} ms{COLOR_RESET}",
            )

