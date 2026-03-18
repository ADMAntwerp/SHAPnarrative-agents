# logger_utils.py or top of your main file
# import logging

# def setup_logger(log_path="results/warning_logs/extraction_warnings.log"):
#     logger = logging.getLogger("extraction_logger")
#     logger.setLevel(logging.INFO)
#     if not logger.handlers:
#         file_handler = logging.FileHandler(log_path, encoding="utf-8")
#         formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
#         file_handler.setFormatter(formatter)
#         logger.addHandler(file_handler)
#     return logger

# # Initialize once in your experiment startup script
# extraction_logger = setup_logger()
# script/logger_utils.py
import logging
import os

def setup_logger(log_path="results/warning_logs/extraction_warnings.log"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("extraction_logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ✅ Global logger instance
extraction_logger = setup_logger()


# ✅ Helper to add context automatically
def log_with_context(level: str, msg: str, model: str = "", dataset: str = "", instance: int | None = None):
    """
    Add standardized context tags to each log entry.
    """
    context = []
    if model:
        context.append(f"model={model}")
    if dataset:
        context.append(f"dataset={dataset}")
    if instance is not None:
        context.append(f"instance={instance}")
    prefix = f"[{' | '.join(context)}] " if context else ""
    full_msg = prefix + msg

    if level.lower() == "warning":
        extraction_logger.warning(full_msg)
    elif level.lower() == "error":
        extraction_logger.error(full_msg)
    else:
        extraction_logger.info(full_msg)
