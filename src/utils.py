"""Shared helpers: logging, seeding, directory setup."""
import logging
import random
from pathlib import Path

import numpy as np


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a logger that writes formatted messages to stdout.

    Args:
        name: Logger name (use __name__ in calling module).
        level: Logging level (default INFO).

    Returns:
        Configured Logger instance.

    Example:
        >>> log = setup_logging(__name__)
        >>> log.info("Training started")
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def set_seed(seed: int = 42) -> None:
    """Fix all global random seeds for reproducibility.

    Args:
        seed: Integer seed value.

    Example:
        >>> set_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dirs(*paths: Path) -> None:
    """Create directories (and parents) if they do not exist.

    Args:
        *paths: One or more pathlib.Path objects.

    Example:
        >>> ensure_dirs(Path("outputs/models"), Path("outputs/figures"))
    """
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
