from datetime import datetime
import pytz

class DateTimeUtils:
    @staticmethod
    def convert_to_ist(dt: datetime) -> datetime:
        ist = pytz.timezone('Asia/Kolkata')
        if dt.tzinfo is None:
            dt = pytz.UTC.localize(dt)
        return dt.astimezone(ist)

    @staticmethod
    def binance_date_converter_to_ist(string_date: str) -> datetime:
        """
        Convert a date string from Binance format to IST timezone.
        :param string_date: Date string in Binance format (e.g., '2023-10-01 12:00:00')
        :return: DateTime object in IST timezone
        """
        dt = datetime.strptime(string_date, '%Y-%m-%d %H:%M:%S')
        return DateTimeUtils.convert_to_ist(dt)