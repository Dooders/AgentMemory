"""Decorators for memory storage operations.

This module provides decorators for storage operations to implement
consistent error handling, logging, and resilience patterns.
"""

import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

from agent_memory.utils.error_handling import (
    MemoryError,
    Priority,
    RedisTimeoutError,
    RedisUnavailableError,
    SQLiteTemporaryError,
    SQLitePermanentError,
)

logger = logging.getLogger(__name__)

# Type variable for the return type of decorated functions
R = TypeVar('R')
F = TypeVar('F', bound=Callable[..., Any])


def log_operation(operation_name: str):
    """Decorator to log memory operations.
    
    Args:
        operation_name: Name of the operation for logs
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> R:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = (time.time() - start_time) * 1000  # ms
                logger.debug(
                    f"{operation_name} completed in {duration:.2f}ms"
                )
                return result
            except Exception as e:
                duration = (time.time() - start_time) * 1000  # ms
                logger.error(
                    f"{operation_name} failed after {duration:.2f}ms: {str(e)}"
                )
                raise
        return wrapper
    return decorator


def retry_on_redis_error(max_retries: int = 3, retry_delay: float = 0.5):
    """Decorator to retry operations on Redis errors.
    
    Args:
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> R:
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (RedisTimeoutError, RedisUnavailableError) as e:
                    last_error = e
                    if attempt < max_retries:
                        delay = retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"Redis error, retrying in {delay:.2f}s ({attempt+1}/{max_retries}): {str(e)}"
                        )
                        time.sleep(delay)
            
            # If we reach here, all retries failed
            if last_error:
                raise last_error
            raise RedisUnavailableError("Maximum retries exceeded")
        return wrapper
    return decorator


def retry_on_sqlite_error(max_retries: int = 3, retry_delay: float = 0.5):
    """Decorator to retry operations on temporary SQLite errors.
    
    Args:
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> R:
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except SQLiteTemporaryError as e:
                    last_error = e
                    if attempt < max_retries:
                        delay = retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"SQLite temporary error, retrying in {delay:.2f}s ({attempt+1}/{max_retries}): {str(e)}"
                        )
                        time.sleep(delay)
                except SQLitePermanentError:
                    # Don't retry permanent errors
                    raise
                    
            # If we reach here, all retries failed
            if last_error:
                raise last_error
            raise SQLiteTemporaryError("Maximum retries exceeded")
        return wrapper
    return decorator


def measure_performance():
    """Decorator to measure and log performance of storage operations.
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> R:
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = (time.time() - start_time) * 1000  # ms
            
            # Get operation name and class name for better logging
            class_name = args[0].__class__.__name__ if args else ""
            op_name = func.__name__
            
            # Log performance data
            if duration > 100:  # Only log slow operations
                logger.warning(f"{class_name}.{op_name} took {duration:.2f}ms")
            else:
                logger.debug(f"{class_name}.{op_name} took {duration:.2f}ms")
                
            return result
        return wrapper
    return decorator


def prioritized_operation():
    """Decorator to handle operation priority.
    
    This decorator expects the decorated function to have a 'priority' parameter,
    and will handle operations differently based on priority level.
    
    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Extract priority from kwargs or use default
            priority = kwargs.get('priority', Priority.NORMAL)
            
            # Critical operations get more retries
            if priority == Priority.CRITICAL:
                # For critical operations, retry more aggressively
                if 'max_retries' in kwargs:
                    kwargs['max_retries'] = max(kwargs['max_retries'], 5)
                else:
                    kwargs['max_retries'] = 5
                    
            # Low priority operations can be throttled during high load
            if priority == Priority.LOW:
                # Add a small delay for low priority operations
                # This could be adjusted based on system load
                time.sleep(0.01)
                
            return func(*args, **kwargs)
        return cast(F, wrapper)
    return decorator 