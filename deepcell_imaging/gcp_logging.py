import logging

from json_log_formatter import JSONFormatter


class GCPFormatter(JSONFormatter):
    """
    A JSON formatter that translates python logs to JSON.

    It fills in the "severity" field from the log level.
    """

    def json_record(self, message: str, extra: dict, record: logging.LogRecord) -> dict:
        extra["severity"] = record.levelname
        return super(GCPFormatter, self).json_record(message, extra, record)


def add_gcp_logging_handler(logger: logging.Logger):
    """
    Add a logging handler to the logger that formats logs as JSON.
    """
    handler = logging.StreamHandler()
    formatter = GCPFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def initialize_gcp_logging():
    """
    Initialize logging for GCP.

    Removes any handlers on the root logger, and
    adds a new handler that formats logs as JSON.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    add_gcp_logging_handler(logger)
    return logger
