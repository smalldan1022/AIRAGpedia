import logging
import sys
from pathlib import Path


class LoggerFactory:
    DEFAULT_FORMAT = "[%(levelname)-5s][%(name)-1s][%(asctime)s] %(message)s (%(filename)s:%(lineno)d)"
    DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    DEFAULT_HANDLER_TYPE = "stream"
    DEFAULT_LEVEL = "INFO"

    def __init__(
        self,
        level: str = "INFO",
        format: str = "",
        date_format: str = "",
        handler_type: str = "stream",
        log_file: str = "",
    ):
        self._level = level or self.DEFAULT_LEVEL
        self._format = format or self.DEFAULT_FORMAT
        self._date_format = date_format or self.DEFAULT_DATE_FORMAT
        self._handler_type = handler_type or self.DEFAULT_HANDLER_TYPE
        self._log_file = log_file

    @property
    def formatter(self) -> logging.Formatter:
        return logging.Formatter(
            fmt=self._format,
            datefmt=self._date_format,
        )

    def _build_handler(self) -> logging.Handler:
        handlers = {
            "stream": self._build_stream_handler,
            "file": self._build_file_handler,
        }

        if self._handler_type not in handlers:
            raise ValueError(
                f"Unknown handler_type: '{self._handler_type}', "
                f"expected one of {list(handlers.keys())}"
            )

        return handlers[self._handler_type]()

    def _build_stream_handler(self) -> logging.StreamHandler:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(self._level)
        handler.setFormatter(self.formatter)
        return handler

    def _build_file_handler(self) -> logging.FileHandler:
        if not self._log_file:
            raise ValueError("log_file must be provided for file handler")
        Path(self._log_file).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(self._log_file)
        handler.setLevel(self._level)
        handler.setFormatter(self.formatter)
        return handler

    def get_logger(self, name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            logger.setLevel(self._level)
            logger.addHandler(self._build_handler())
        return logger


if __name__ == "__main__":
    logger = LoggerFactory(level="INFO").get_logger(name=__name__)
    logger.info("test")
