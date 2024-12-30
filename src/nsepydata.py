from datetime import datetime
from io import StringIO
import logging
import pandas as pd
import re
import requests
from typing import Tuple

class NSEPyData():
    def __init__(self):
        self.__logger = logging.getLogger(__name__)
        self.__logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
        fileHandler = logging.FileHandler('./download-log.txt', mode='a')
        consoleHandler = logging.StreamHandler()
        fileHandler.setFormatter(formatter)
        consoleHandler.setFormatter(formatter)
        logging.getLogger('').addHandler(consoleHandler)
        logging.getLogger('').addHandler(fileHandler)

        self.headers = {
            "Host": "www.nseindia.com",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
        }


    def __download_historical_price_volume(self, symbol, series, start=None, end=None):
        last = False
        url = "https://www.nseindia.com/api/historical/cm/equity"

        # Initialize the final DataFrame
        nse_df = pd.DataFrame()

        # Session to manage cookies automatically
        with requests.Session() as session:
            # Send an initial request to the main site to get cookies and headers
            session.get('https://www.nseindia.com', headers=self.headers)

            self.__logger.info(f"Fetching data for {symbol} from {start.strftime('%d-%b-%Y')} to {end.strftime('%d-%b-%Y')}.")
            while not last:
                begin = end.replace(month=1, day=1)
                if start != None and begin <= start:
                    begin = start
                    last = True

                # Update request parameters
                params = {
                    "symbol": symbol,
                    "series": f'["{series}"]',
                    "from": begin.strftime("%d-%m-%Y"),
                    "to": end.strftime("%d-%m-%Y"),
                    "csv": "true",
                }

                # Make the actual request to download the CSV file
                response = session.get(url, headers=self.headers, params=params)
                
                # Check if response is empty
                if not response.content.strip():
                    self.__logger.info("No more data to fetch.")
                    break

                # Parse the CSV content and append to the final DataFrame
                try:
                    df = pd.read_csv(StringIO(response.text))
                    if df.shape[0] != 0:
                        nse_df = pd.concat([nse_df, df], ignore_index=True)
                        self.__logger.info(f"Fetched data from {start.strftime('%d-%m-%Y')} to {end.strftime('%d-%m-%Y')}.")
                    else:
                        self.__logger.info("No more data to fetch.")
                        break
                except Exception as e:
                    self.__logger.error(f"Error parsing response for {start.year}: {e}")
                    break

                # Move to the previous year
                end = end.replace(month=12, day=31, year=end.year - 1)

        nse_df.columns.values[0] = 'DATE'
        nse_df.columns.values[2] = 'OPEN'
        nse_df.columns.values[3] = 'HIGH'
        nse_df.columns.values[4] = 'LOW'
        nse_df.columns.values[7] = 'CLOSE'
        nse_df.columns.values[11] = 'VOLUME'
        return nse_df           


    def __download_all_corporate_actions(self):
        url = "https://www.nseindia.com/api/corporates-corporateActions?index=equities&from_date=05-01-2024&to_date=05-12-2024&csv=true"


    def __download_corporate_action(self, symbol):
        url = "https://www.nseindia.com/api/corporates-corporateActions"

        # Initialize the final DataFrame
        nse_df = pd.DataFrame()

        # Session to manage cookies automatically
        with requests.Session() as session:
            # Send an initial request to the main site to get cookies and headers
            session.get('https://www.nseindia.com', headers=self.headers)

            # Update request parameters
            params = {
                "index": "equities",
                "symbol": symbol,
                "csv": "true"
            }

            # Make the actual request to download the CSV file
            response = session.get(url, headers=self.headers, params=params)
            
            # Check if response is empty
            if not response.content.strip():
                self.__logger.info("No more data to fetch.")

            # Parse the CSV content and append to the final DataFrame
            try:
                df = pd.read_csv(StringIO(response.text))
                if df.shape[0] != 0:
                    nse_df = pd.concat([nse_df, df], ignore_index=True)
                    self.__logger.info(f"Fetched coporate action data for {symbol}.")
                else:
                    self.__logger.info("No more data to fetch.")
            except Exception as e:
                self.__logger.error(f"Error parsing response: {e}")

        return nse_df       


    def __adjust_for_corp_action(self, ohlcv_df, corp_act_df):    
        # Filter rows containing the words "Split" or "Bonus"
        filt_corp_act_df = corp_act_df[corp_act_df['PURPOSE'].str.contains('Split|Bonus', case=False, na=False)]
        ohlcv_df['DATE'] = pd.to_datetime(ohlcv_df['DATE'], format='%d-%b-%Y')
        start = end = None
        adj_columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
        adj_nse_df = pd.DataFrame()
        for index, row in filt_corp_act_df.iterrows():
            record_date = row['RECORD DATE']
            purpose = row['PURPOSE']
            pattern = r"[^0-9]+(\d+)[^0-9]+(\d+)"
            matches = re.search(pattern, purpose)
            if matches:
                try:
                    factor = float(matches.group(1)) / float(matches.group(2))
                except ZeroDivisionError:
                    raise ValueError("Division by zero in corporate action factor calculation.")

                factor = int(matches.group(1)) / int(matches.group(2))
                start = record_date
                if end is None:
                    filt_ohlcv_df = ohlcv_df[(ohlcv_df['DATE'] >= start)]
                else:
                    filt_ohlcv_df = ohlcv_df[(ohlcv_df['DATE'] >= start) & (ohlcv_df['DATE'] < end)]
                    filt_ohlcv_df[adj_columns] = (filt_ohlcv_df[adj_columns] / factor).round(2)
                    filt_ohlcv_df['VOLUME'] *= int(factor)
                end = start
                adj_nse_df = pd.concat([adj_nse_df, filt_ohlcv_df], ignore_index=True)
        
        if start is not None:
            filt_ohlcv_df = ohlcv_df[(ohlcv_df['DATE'] < start)]
            filt_ohlcv_df[adj_columns] = (filt_ohlcv_df[adj_columns] / factor).round(2)
            filt_ohlcv_df['VOLUME'] *= int(factor)
            adj_nse_df = pd.concat([adj_nse_df, filt_ohlcv_df], ignore_index=True)

        return adj_nse_df


    def __extractOHLCV(self, nse_df):
        columns = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
        OHLCV_df = nse_df[columns]
        OHLCV_df.rename(columns={'Date': 'DATE'}, inplace=True)
        OHLCV_df.rename(columns={'close': 'CLOSE'}, inplace=True)
        OHLCV_df[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']] = OHLCV_df[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']].replace({',': ''}, regex=True).astype(float)
        OHLCV_df[['VOLUME']] = OHLCV_df[['VOLUME']].replace({',': ''}, regex=True).astype(int)
        return OHLCV_df


    def change_time_period(self, nse_ohlcv_df: pd.DataFrame, timeperiod: str) -> pd.DataFrame:
        """
        Adjusts the time period of the OHLCV data.

        Args:
            nse_ohlcv_df (pd.DataFrame): DataFrame containing daily OHLCV data with columns:
                                        'DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'.
            timeperiod (str): String specifying the aggregation period in the format <NUMBER><LETTER>.
                            LETTER can be 'D', 'W', 'M', 'Q', 'Y' for Daily, Weekly, Monthly, Quarterly, Yearly.
                            NUMBER specifies the number of periods (e.g., '2D' means 2 Days).

        Returns:
            pd.DataFrame: Aggregated DataFrame based on the specified time period.
        """
        # Extract number and letter from timeperiod
        num = int(''.join(filter(str.isdigit, timeperiod)))
        period_type = ''.join(filter(str.isalpha, timeperiod)).upper()

        # Ensure 'DATE' column is a datetime object
        nse_ohlcv_df['DATE'] = pd.to_datetime(nse_ohlcv_df['DATE'], format='%m/%d/%Y')

        # Set 'DATE' as the index for easier resampling
        nse_ohlcv_df = nse_ohlcv_df.set_index('DATE')

        # Define the resampling rule based on period type
        if period_type == 'D':  # Daily aggregation
            rule = f'{num}D'
        elif period_type == 'W':  # Weekly aggregation
            rule = f'{num}W'
        elif period_type == 'M':  # Monthly aggregation
            rule = f'{num}M'
        elif period_type == 'Q':  # Quarterly aggregation (starting Jan, Apr, Jul, Oct)
            rule = f'{num}Q'
        elif period_type == 'Y':  # Yearly aggregation
            rule = f'{num}A'
        else:
            raise ValueError(f"Invalid period type: {period_type}. Must be one of 'D', 'W', 'M', 'Q', 'Y'.")

        # Perform aggregation
        def aggregate_period(group):
            return pd.Series({
                'DATE': group.index.min(),  # First date of the period
                'OPEN': group['OPEN'].iloc[0],  # First value of the period
                'HIGH': group['HIGH'].max(),   # Maximum value of the period
                'LOW': group['LOW'].min(),     # Minimum value of the period
                'CLOSE': group['CLOSE'].iloc[-1],  # Last value of the period
                'VOLUME': group['VOLUME'].sum()   # Sum of values
            })

        adj_nse_df = nse_ohlcv_df.resample(rule).apply(aggregate_period).dropna()

        # Reset index to bring 'DATE' back as a column
        adj_nse_df = adj_nse_df.reset_index(drop=True)
        # Sort the DataFrame by 'DATE' in descending order
        adj_nse_df = adj_nse_df.sort_values(by='DATE', ascending=False).reset_index(drop=True)
        # Format the 'DATE' column back to mm/dd/yyyy format
        adj_nse_df['DATE'] = adj_nse_df['DATE'].dt.strftime('%m/%d/%Y')

        return adj_nse_df


    def get_OHLCV_data(self, symbol: str, series: str, start: datetime, end:datetime=None, adjust_corp_action:bool=True, timeperiod:str='1D') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Downloads the historical data of a NSE symbol.

        Args:
            symbol (str): [Required] NSE symbol for which the historical data needs to be downloaded
            start (datetime): [Required] A datetime object indicating from when the historical data is needed
            end (datetime): [Optional] A datetime object indicating until when the historical data is needed. Default: Today's date
            adjust_corp_action (bool): [Optional] A boolean variable indicating if the stock price should be adjusted for corporate actions. Default = True
            timeperiod (str) : [Optional] Aggregate stock quote so that every row in the dataframe corresponds to this duration - 1W, 2W, 1M, 1Q, 1Y. Default = '1D'

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 
            - The first DataFrame contains OHLC (Open, High, Low, Close) data.
            - The second DataFrame contains corporate action data.
            If no data is found, both DataFrames will be empty.
        """

        end = datetime.now().date() if end == None else datetime.strptime(end, '%d-%b-%Y')
        start = datetime.strptime(start, '%d-%b-%Y')
        nse_df = self.__download_historical_price_volume(symbol, series, start, end)
        nse_ohlcv_df = self.__extractOHLCV(nse_df)
        nse_corp_act_df = pd.DataFrame()
        
        if adjust_corp_action:
            nse_corp_act_df = self.__download_corporate_action(symbol)
            nse_ohlcv_df = self.__adjust_for_corp_action(nse_ohlcv_df, nse_corp_act_df)
        
        timeperiod = timeperiod.upper()
        if timeperiod != '1D':
            nse_ohlcv_df = self.change_time_period(nse_ohlcv_df, timeperiod)

        return nse_ohlcv_df, nse_corp_act_df


    def download_bhav(self, date):
            """
            Downloads the BhavCopy file for a given date and returns its content.

            Args:
                date (str): Date in the format "DD-MMM-YYYY".

            Returns:
                bytes: Content of the downloaded file.
            """
            date_download = datetime.strptime(date, "%d-%b-%Y").strftime("%Y%m%d")
            url = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_<DATE>_F_0000.csv.zip"
            url = re.sub('<DATE>', date_download, url)

            with requests.Session() as session:
                try:
                    # Send an initial request to the main site to get cookies and headers
                    session.get('https://www.nseindia.com', headers=self.headers)   

                    # Send a GET request
                    self.headers['Host'] = 'nsearchives.nseindia.com'
                    response = session.get(url, headers=self.headers)
                    response.raise_for_status()  # Raise exception for HTTP errors
                    
                    # Return the content of the response
                    self.__logger.info(f"File downloaded successfully from URL: {url}")
                    return response.content
                except requests.exceptions.RequestException as e:
                    self.__logger.error(f"Error downloading file: {e}")
                    raise


    def download_index_bhav(self):
        url = "https://www.niftyindices.com/Daily_Snapshot/ind_close_all_13122024.csv"


    def download_nse_index_constitutents(self, index):
        url = {"NIFTY750": "https://www.niftyindices.com/IndexConstituent/ind_niftytotalmarket_list.csv",
               "NIFTY50": "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv"}

        with requests.Session() as session:
            try:
                # Send an initial request to the main site to get cookies and headers
                session.get('https://www.niftyindices.com', headers=self.headers)   

                # Send a GET request
                self.headers['Host'] = 'niftyindices.com'
                response = session.get(url[index], headers=self.headers)
                response.raise_for_status()  # Raise exception for HTTP errors
                
                # Return the content of the response
                self.__logger.info(f"File downloaded successfully from URL: {url}")
                return response.content
            except requests.exceptions.RequestException as e:
                self.__logger.error(f"Error downloading file: {e}")
                raise


