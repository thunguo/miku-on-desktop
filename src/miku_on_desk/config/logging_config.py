"""日志系统初始化：统一配置根 logger，业务模块通过 ``logging.getLogger(__name__)`` 取子 logger。"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 单文件 10MB，保留 5 个历史文件，足够覆盖排障窗口且不无限增长。
_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 5


def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    """配置根 logger：控制台 handler + 按大小轮转的文件 handler。

    必须在应用启动时调用且只调用一次；重复调用会导致 handler 重复叠加，
    因此这里先清空已有 handler 再重新装配。
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(stream=sys.stderr)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        filename=log_dir / "miku-on-desk.log",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
