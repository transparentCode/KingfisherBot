def get_price_source_data(data, source):
    """
    Calculate price source data based on the specified source type.

    :param data: DataFrame containing OHLC price data
    :param source: Source type ('close', 'hlc3', 'hl2', 'ohlc4', 'hlcc4', or column name)
    :return: Series with calculated source data
    """
    if source == 'hlcc4':
        return (data['high'] + data['low'] + data['close'] + data['close']) / 4
    elif source == 'hlc3':
        return (data['high'] + data['low'] + data['close']) / 3
    elif source == 'hl2':
        return (data['high'] + data['low']) / 2
    elif source == 'ohlc4':
        return (data['open'] + data['high'] + data['low'] + data['close']) / 4
    elif source in data.columns:
        return data[source]
    else:
        raise ValueError(f"Invalid source '{source}'. Must be a column name or a valid combination.")


def get_available_price_sources():
    """Return list of available price source options."""
    return ['close', 'open', 'high', 'low', 'hlc3', 'hl2', 'ohlc4', 'hlcc4']
