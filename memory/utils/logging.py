import os
import logging
import sys

# Use current working directory for logs
LOGS_DIR = os.path.join(os.getcwd(), "logs")

print(f"Current working directory: {os.getcwd()}")
print(f"Logs directory: {LOGS_DIR}")


def setup_logging(demo_name: str) -> logging.Logger:
    """Set up logging to both console and file.

    Args:
        demo_name: Name of the demo for log file naming

    Returns:
        Configured logger instance
    """
    try:
        # Create logs directory if it doesn't exist
        os.makedirs(LOGS_DIR, exist_ok=True)
        print(f"Created logs directory at: {LOGS_DIR}")

        # Use a fixed log filename based on demo name (without timestamp)
        log_file = os.path.join(LOGS_DIR, f"{demo_name}.log")
        print(f"Log file will be created at: {log_file}")

        # Clear the existing log file if it exists
        with open(log_file, "w") as f:
            # Empty the file by opening it in write mode
            pass
        print(f"Initialized log file: {log_file}")

        # Configure logging
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        # Clear existing handlers if any
        if logger.handlers:
            logger.handlers.clear()

        # File handler for logging to file
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)

        # Console handler for logging to console
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)

        # Create a filter to exclude embedding-related log messages
        class EmbeddingFilter(logging.Filter):
            def filter(self, record):
                # Skip any log messages containing "embedding" or "vector"
                return not any(
                    term in record.getMessage().lower()
                    for term in ["embedding", "vector", "encoded"]
                )

        # Add the filter to both handlers
        embedding_filter = EmbeddingFilter()
        file_handler.addFilter(embedding_filter)
        console_handler.addFilter(embedding_filter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Log a test message to verify logging is working
        logger.info(f"Logging initialized for {demo_name}")
        print(f"Logging setup complete for {demo_name}")

        return logger
    except Exception as e:
        print(f"Error setting up logging: {e}")
        # Fallback to basic console logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger()