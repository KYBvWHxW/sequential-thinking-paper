import logging
import functools
import time
import json
from typing import Any, Callable

class StructuredLogger(logging.Logger):
    def __init__(self, name: str):
        super().__init__(name)

    def _log_structured(self, level: int, msg: str, event: str = None, details: dict = None, **kwargs):
        if details is None:
            details = {}
        
        if isinstance(msg, (list, tuple)) and len(msg) > 0:
            msg = msg[0] if isinstance(msg[0], str) else str(msg[0])
        
        if kwargs.get('extra') is None:
            kwargs['extra'] = {}
        
        kwargs['extra'].update({
            "event": event,
            "details": details
        })
        
        super().log(level, msg, **kwargs)

    def info(self, msg, *args, event: str = None, details: dict = None, **kwargs):
        if args:
            msg = msg % args
        self._log_structured(logging.INFO, msg, event, details, **kwargs)

    def error(self, msg, *args, event: str = None, details: dict = None, **kwargs):
        if args:
            msg = msg % args
        self._log_structured(logging.ERROR, msg, event, details, **kwargs)

    def debug(self, msg, *args, event: str = None, details: dict = None, **kwargs):
        if args:
            msg = msg % args
        self._log_structured(logging.DEBUG, msg, event, details, **kwargs)

    def warning(self, msg, *args, event: str = None, details: dict = None, **kwargs):
        if args:
            msg = msg % args
        self._log_structured(logging.WARNING, msg, event, details, **kwargs)

# Register our custom logger class
logging.setLoggerClass(StructuredLogger)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with a higher log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)

def with_logging(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to add logging to functions."""
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.info(
                f"{func.__name__} completed",
                event="function_complete",
                details={"elapsed_time": f"{elapsed_time:.2f}s"}
            )
            return result
        except Exception as e:
            logger.error(
                f"Error in {func.__name__}",
                event="function_error",
                details={"error": str(e)}
            )
            raise
    return wrapper
