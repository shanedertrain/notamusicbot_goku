import logging
from pathlib import Path

DEBUG = True  # Set to True, mostly just disables the opening message to not be annoying

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
DATETIME_FORMAT_FILESAFE = "%Y%m%d_%H%M%S"

FOLDER_ROOT = Path(__file__).parent

FOLDER_INPUT = FOLDER_ROOT / "input"
FOLDER_INPUT.mkdir(parents=True, exist_ok=True)

FILEPATH_USERS = FOLDER_INPUT / "users.json"

FOLDER_OUTPUT = FOLDER_ROOT / "output"
FOLDER_OUTPUT.mkdir(parents=True, exist_ok=True)

FOLDER_TTS = FOLDER_OUTPUT / "tts"
FOLDER_TTS.mkdir(parents=True, exist_ok=True)

FOLDER_LOGS = FOLDER_ROOT / "logs"
FOLDER_LOGS.mkdir(parents=True, exist_ok=True)

FILEPATH_LOG = FOLDER_LOGS / "log.log"
FILEPATH_COMBINED_LOG = FOLDER_LOGS / "combined.log"


def configure_logger(log_file) -> logging.Logger:
    logger = logging.getLogger("logger_rvc")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Avoid duplicate handlers if configure_logger is called multiple times.
    if logger.handlers:
        logger.handlers.clear()

    # Create file handler which logs only messages above INFO level to the file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create a formatter and set it to the handler
    # Include filename and line number
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s: %(filename)s:%(lineno)d | %(message)s", "%H:%M:%S")
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)

    return logger


# Configure root logging so third-party loggers (e.g., discord) also land in a file.
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(FILEPATH_COMBINED_LOG, mode="a", encoding="utf-8")
    ],
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True,
)

LOGGER = configure_logger(FILEPATH_LOG)
