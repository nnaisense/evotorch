import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

_formatter = logging.Formatter("[%(asctime)s] <pid:%(process)d> %(pathname)s:%(lineno)d: %(levelname)s: %(message)s")

_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.addFilter(lambda log_record: log_record.levelno < logging.WARNING)
_stdout_handler.setFormatter(_formatter)
logger.addHandler(_stdout_handler)

_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.addFilter(lambda log_record: log_record.levelno >= logging.WARNING)
_stderr_handler.setFormatter(_formatter)
logger.addHandler(_stderr_handler)
