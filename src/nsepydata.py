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
        self.equity_df = self.get_securities_in_segment('EQUITY')
        self.sme_df = self.get_securities_in_segment('SME')


    def __download_csv(self, url, host=None, params=None):
        # Initialize the final DataFrame
        nse_df = None

        # Session to manage cookies automatically
        with requests.Session() as session:
            headers = {
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
            if host is not None:
                headers['Host'] = host
            # Send an initial request to the main site to get cookies and headers
            session.get('https://'+host, headers=headers)
            # Make the actual request to download the CSV file
            response = session.get(url, headers=headers, params=params)
            
            # If response is not empty, read the csv file
            if response.content.strip():
                try:
                    nse_df = pd.read_csv(StringIO(response.text))
                except Exception as e:
                    self.__logger.error(f"Error parsing response: {e}")
        return nse_df  


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


    def __get_action_factor(self, purpose):
        actions = {"split": "SPLIT", "bonus": "BONUS"}
        action = None
        purpose = purpose.lower()
        for key, value in actions.items():
            if key in purpose:
                action = value

        pattern = r"[^0-9]+(\d+)[^0-9]+(\d+)"
        matches = re.search(pattern, purpose)
        if matches:
            if action == 'SPLIT':
                try:
                    factor = float(matches.group(1)) / float(matches.group(2))
                except ZeroDivisionError:
                    raise ValueError("Division by zero in corporate action factor calculation.")
            elif action == 'BONUS':
                try:
                    factor = (float(matches.group(2)) + float(matches.group(1))) / float(matches.group(2))
                except ZeroDivisionError:
                    raise ValueError("Division by zero in corporate action factor calculation.")                
        return action, factor


    def __adjust_for_corp_action(self, ohlcv_df, corp_act_df):   
        if not corp_act_df.empty:
            # Filter rows containing the words "Split" or "Bonus"
            filt_corp_act_df = corp_act_df[corp_act_df['PURPOSE'].str.contains('Split|Bonus', case=False, na=False)]
            start = None
            adj_columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
            for index, row in filt_corp_act_df.iterrows():
                record_date = row['RECORD DATE']
                purpose = row['PURPOSE']
                action, factor = self.__get_action_factor(purpose)
                if action is not None:
                    start = datetime.strptime(record_date, '%d-%b-%Y')
                    ohlcv_df.loc[(ohlcv_df['DATE'] < start), adj_columns] = (ohlcv_df[adj_columns] / factor).round(2)
                    ohlcv_df.loc[(ohlcv_df['DATE'] < start), 'VOLUME'] = (ohlcv_df['VOLUME'] * factor).astype(int)

        return ohlcv_df


    def __extractOHLCV(self, nse_df):
        columns = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
        OHLCV_df = nse_df[columns]
        OHLCV_df = OHLCV_df.rename(columns={'Date': 'DATE', 'close': 'CLOSE'})
        OHLCV_df.loc[:, ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']] = OHLCV_df[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']].replace({',': ''}, regex=True).astype(float)
        OHLCV_df.loc[:, ['VOLUME']] = OHLCV_df[['VOLUME']].replace({',': ''}, regex=True).astype(int)
        OHLCV_df.loc[:, 'DATE'] = pd.to_datetime(OHLCV_df['DATE'], format='%d-%b-%Y')

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
            if group.empty:
                return None  # Return None for empty groups
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

        return adj_nse_df


    def get_corporate_action_data(self, symbol):
        """
        Returns all corporate actions of a NSE traded stock as a pandas dataframe

        Args:
            symbol (str): [Required] NSE symbol for which corporate action is required

        Returns:
            pd.DataFrame: 
            - The dataframe contains the following columns - 'DATE' 'OPEN' 'HIGH' 'LOW' 'CLOSE' and 'VOLUME'. The dataframe can be indexed using the 'DATE' column.
            Every row of the dataframe refers to 1 timeperiod worth of data and the date refers to the start of the period
            Returns None, if no data is found
        """        
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


    def get_OHLCV_data(self, symbol: str, start: str, end:str=None, 
                       adjust_corp_action:bool=True, timeperiod:str='1D') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns the historical data of a NSE traded stock as a pandas dataframe

        Args:
            symbol (str): [Required] NSE symbol for which the historical data needs to be downloaded
            start (str): [Required] Start date in the form of dd-mmm-yyyy indicating from when the historical data is needed. Example: '15-Sep-2008'
            end (str): [Optional] End date in the form of dd-mmm-yyyy indicating until when the data is needed. Example: '15-Sep-2008'. Default: Today's date
            adjust_corp_action (bool): [Optional] A boolean variable indicating if the stock price should be adjusted for corporate actions (bonus and stock splits). Default = True
            timeperiod (str) : [Optional] Aggregate stock quote so that every row in the dataframe corresponds to this duration - 1W, 2W, 1M, 1Q, 1Y. Default = '1D'

        Returns:
            pd.DataFrame: 
            - The dataframe contains the following columns - 'DATE' 'OPEN' 'HIGH' 'LOW' 'CLOSE' and 'VOLUME'. The dataframe can be indexed using the 'DATE' column.
            Every row of the dataframe refers to 1 timeperiod worth of data and the date refers to the start of the period
            Returns None, if no data is found
        """
        series = nse_ohlcv_df = None
        symbol = symbol.upper()
        if symbol in self.equity_df['SYMBOL'].values:
            series = self.equity_df.loc[self.equity_df['SYMBOL'] == symbol, 'SERIES'].iloc[0]
        elif symbol in self.sme_df['SYMBOL'].values:
            series = self.sme_df.loc[self.sme_df['SYMBOL'] == symbol, 'SERIES'].iloc[0]

        if series is not None:
            fetch_end = datetime.now() if end == None or adjust_corp_action else datetime.strptime(end, '%d-%b-%Y')
            start = datetime.strptime(start, '%d-%b-%Y')
            nse_df = self.__download_historical_price_volume(symbol, series, start, fetch_end)
            nse_ohlcv_df = self.__extractOHLCV(nse_df)
            
            if adjust_corp_action:
                nse_corp_act_df = self.get_corporate_action_data(symbol)
                nse_ohlcv_df = self.__adjust_for_corp_action(nse_ohlcv_df, nse_corp_act_df)

            if end is not None:
                end = datetime.strptime(end, '%d-%b-%Y')
                nse_ohlcv_df = nse_ohlcv_df[(nse_ohlcv_df['DATE'] <= end)]

            timeperiod = timeperiod.upper()
            if timeperiod != '1D':
                nse_ohlcv_df = self.change_time_period(nse_ohlcv_df, timeperiod)
            
            nse_ohlcv_df['DATE'] = pd.to_datetime(nse_ohlcv_df['DATE']).dt.strftime('%d-%b-%Y')

        return nse_ohlcv_df


    def get_securities_in_segment(self, segment) -> pd.DataFrame:
        """
        Returns the list of NSE traded stocks in the EQUITY or SME segment as a pandas dataframe.

        Args:
            segment (str): [Required] 'EQUITY' / 'SME'

        Returns:
            pd.DataFrame: 
            - Contains the list of NSE traded stocks in the provided segment
            The dataframe contains the following columns 'SYMBOL', 'NAME_OF_COMPANY', 'SERIES', 'DATE_OF_LISTING', 'PAID_UP_VALUE', 'ISIN_NUMBER', 'FACE_VALUE'. 
            An additional column by the name 'MARKET_LOT' is also available in case the segment is 'EQUITY'
            Returns None, if no data is found
        """        
        segment = segment.upper()
        host = 'nsearchives.nseindia.com'
        if segment == 'EQUITY':
            url = 'https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv'
        elif segment == 'SME':
            url = 'https://nsearchives.nseindia.com/emerge/corporates/content/SME_EQUITY_L.csv'

        securities_df = self.__download_csv(url, host=host)

        if segment == 'EQUITY':
            securities_df = securities_df.rename(columns={'NAME OF COMPANY': 'NAME_OF_COMPANY', ' SERIES': 'SERIES', 
                                                          ' DATE OF LISTING': 'DATE_OF_LISTING', ' PAID UP VALUE': 'PAID_UP_VALUE',
                                                          ' MARKET LOT': 'MARKET_LOT', ' ISIN NUMBER': 'ISIN_NUMBER', ' FACE VALUE': 'FACE_VALUE', })

        return securities_df


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


    def __download_all_corporate_actions(self):
        url = "https://www.nseindia.com/api/corporates-corporateActions?index=equities&from_date=05-01-2024&to_date=05-12-2024&csv=true"

