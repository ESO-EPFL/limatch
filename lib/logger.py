from pathlib import Path
import logging
logger = logging.getLogger("LiMatch")

def get_logger(cfg):
    logger = logging.getLogger("LiMatch")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger 

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if cfg.get("log_file", None):
        Path(cfg["log_file"]).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(cfg["log_file"])
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger

def stage(logger, name, msg):
    logger.info(f"[{name}] {msg}")

def log_progress(logger, step, label):
    logger.info(f"[{step}/7] {label}")

def log_stage(logger, msg):
    """Pipeline-level stage header"""
    logger.info(msg)

def log_sub(logger, msg):
    """Indented log for inside-stage messages"""
    logger.info(f"  └─ {msg}")

def log_sub_sub(logger, msg):
    """Further indented log for inside-substage messages"""
    logger.info(f"      └─ {msg}")