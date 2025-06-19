import pandas as pd
import pytz


class Data_Utils:

    @staticmethod
    def transform_data(data):
        df = pd.DataFrame(
            data.get('candles', []),
            columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        )

        df['Timestamp'] = (pd.to_datetime(df['Timestamp'], unit='s')
                           .dt
                           .tz_localize('UTC')
                           .dt
                           .tz_convert(pytz.timezone('Asia/Kolkata')))
        # df.set_index('Timestamp', inplace=True)

        return df