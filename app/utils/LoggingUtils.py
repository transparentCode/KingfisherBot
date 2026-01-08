import logging
import datetime

import pytz


class LoggerFormatter(logging.Formatter):
    """
    Custom formatter for logs that adds colors to console output and
    provides additional formatting options.
    """
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',  # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[91m\033[1m',  # Bold Red
        'RESET': '\033[0m'  # Reset to default
    }

    def format(self, record):
        """
        Format the specified record as text.

        Adds color to console output and additional context if available.
        """
        # Apply the base formatting from parent class
        log_message = super().format(record)

        # Add color for console output - will be ignored in files
        level_name = record.levelname
        if level_name in self.COLORS:
            log_message = f"{self.COLORS[level_name]}{log_message}{self.COLORS['RESET']}"

        # Add extra fields if they exist
        if hasattr(record, 'extra_data'):
            log_message += f" | Extra: {record.extra_data}"

        return log_message

    def formatTime(self, record, datefmt=None):
        # Convert timestamp to datetime in UTC
        utc_dt = datetime.datetime.fromtimestamp(record.created, tz=pytz.UTC)

        # Convert to IST
        ist_tz = pytz.timezone('Asia/Kolkata')
        ist_dt = utc_dt.astimezone(ist_tz)

        if datefmt:
            return ist_dt.strftime(datefmt)
        else:
            return ist_dt.isoformat()


class FileLoggerFormatter(LoggerFormatter):
    """
    Formatter for file logs that does NOT add ANSI colors but keeps the custom time formatting.
    """
    def format(self, record):
        # Call the base logging.Formatter.format, which will use our custom formatTime
        # We bypass LoggerFormatter.format to avoid adding colors
        log_message = logging.Formatter.format(self, record)

        # Add extra fields if they exist
        if hasattr(record, 'extra_data'):
            log_message += f" | Extra: {record.extra_data}"

        return log_message
