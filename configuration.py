from pathlib import Path
import logging

DEBUG = False #mostly just disabled the opening message to not be annoying

DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
DATETIME_FORMAT_FILESAFE = '%Y%m%d_%H%M%S'

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

FILEPATH_LOG = FOLDER_LOGS / 'log.log'

def configure_logger(log_file) -> logging.Logger:
    logger = logging.getLogger('logger_rvc')
    logger.setLevel(logging.DEBUG)

    # Create file handler which logs only messages above INFO level to the file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create stream handler to log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Change the level as needed

    # Create a formatter and set it to the handler
    # Include filename and line number
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s: %(filename)s:%(lineno)d | %(message)s', '%H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

LOGGER = configure_logger(FILEPATH_LOG)