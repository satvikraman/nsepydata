# NSEPyData: An unofficial client (v1.0) to read data off NSE

## Overview

This Python library provides methods to fetch reports and historical data from NSE (National Stock Exchange) in the form of pandas dataframe. The code is available under the MIT license and hence you can use this code any which way you want.

## Usage

### Import the package into your code

```python
from nsepydata import NSEPyData
```

### `get_OHLCV_data`

This function downloads the historical data for an NSE traded stock and optionally adjusts the data for corporate actions like bonus and splits. The output is provided in the form of a pandas dataframe with the following columns DATE, OPEN, HIGH, LOW, CLOSE, VOLUME

#### Arguments

- **`symbol` (str)**: _(Required)_ The NSE symbol for which the historical data needs to be downloaded.
- **`start` (str)**: _(Required)_ Start date in the form of dd-mmm-yyyy indicating from when the historical data is needed
- **`end` (str)**: _(Optional)_ End date in the form of dd-mmm-yyyy indicating until when the data is needed. Defaults to today's date if not provided.
- **`adjust_corp_action` (bool)**: _(Optional)_ Whether to adjust the stock price for corporate actions like bonus and splits. Defaults to `True`.
- **`timeperiod` (str)**: _(Optional)_ The aggregation period for stock quotes. Acceptable values are:
  - `1D`: Daily (default)
  - `1W`: Weekly
  - `2W`: Bi-weekly
  - `1M`: Monthly
  - `1Q`: Quarterly
  - `1Y`: Yearly

#### Returns

A Pandas DataFrame:

1. **OHLC DataFrame**: Contains the OPEN, HIGH, LOW, CLOSE and VOLUME data in columns named the same
   Returns `None`, if no data is found for the specified symbol and date range.

#### Example Usage

```python
from nsepydata import NSEPyData

# Instantiate the class
obj = NSEPyData()

# Get historical data for a symbol
ohlcv_data = obj.get_OHLCV_data(symbol='RELIANCE', start='01-Dec-2023', end='31-Dec-2024', adjust_corp_action=True, timeperiod='1D')
# Display the data
print("OHLC and Volume Data:")
print(ohlcv_data)
```
