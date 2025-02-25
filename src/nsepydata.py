from datetime import datetime, timedelta
from io import StringIO
import io
import logging
import pandas as pd
import re
import requests
import zipfile
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
            response = session.get('https://'+host, headers=headers)
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

            df = self.__download_csv(url, host='www.nseindia.com', params=params)
            
            # Check if response is empty
            if df is None:
                self.__logger.info("No more data to fetch.")
                break

            # Parse the CSV content and append to the final DataFrame
            try:
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
    

    def __get_action_factor(self, purpose, prev_close=None):
        actions = {"split": "SPLIT", "bonus": "BONUS", "dividend": "DIVIDEND"}
        action = None
        factor = 1
        purpose = purpose.lower()
        
        for key, value in actions.items():
            if key in purpose:
                action = value

        pattern = r"[^0-9]+(\d+)[^0-9]+(\d+)" if action in ['SPLIT', 'BONUS'] else r"(\d+(?:\.\d+)?)"
        matches = re.search(pattern, purpose)
        if matches:
            if action == 'SPLIT':
                try:
                    factor = float(matches.group(1)) / float(matches.group(2))
                except ZeroDivisionError:
                    self.__logger.error("Division by zero in corporate action factor calculation.")
            elif action == 'BONUS':
                try:
                    factor = (float(matches.group(2)) + float(matches.group(1))) / float(matches.group(2))
                except ZeroDivisionError:
                    self.__logger.error("Division by zero in corporate action factor calculation.")
            elif action == 'DIVIDEND':
                dividend = float(matches.group(1))
                factor = (prev_close - dividend) / prev_close
        else:
            action = None
            self.__logger.critical("Purpose yielded no adjustment factor %s", purpose)

        return action, factor


    def __adjust_for_split_bonus(self, ohlcv_df:pd.DataFrame, start_date:datetime, end_date:datetime):
        # Filter rows containing the words "Split" or "Bonus"
        filt_corp_act_df = ohlcv_df[['DATE', 'CORP_ACTION']][ohlcv_df['CORP_ACTION'].str.contains('Split|Bonus', case=False, na=False)]
        adj_columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE']

        for index, row in filt_corp_act_df.iterrows():
            ex_date_dt = row['DATE']
            mask = ohlcv_df['DATE'] < ex_date_dt
            if mask.any() and (start_date <= ex_date_dt and ex_date_dt <= end_date):
                purposes = re.findall(r'<(.*?)\/>', row['CORP_ACTION'])  # Split the string and remove empty elements
                for purpose in purposes:
                    if re.search(r'\b(Split|Bonus)\b', purpose):
                        action, factor = self.__get_action_factor(purpose)
                        if action is not None:
                            ohlcv_df.loc[mask, adj_columns] = (ohlcv_df[adj_columns] / factor).round(2)
                            ohlcv_df.loc[mask, 'VOLUME'] = (ohlcv_df['VOLUME'] * factor).astype(int)
            else: 
                break
        return ohlcv_df


    def __adjust_for_div(self, ohlcv_df: pd.DataFrame, start_date: datetime, end_date: datetime):
        # Filter rows containing the word "Dividend" within the specified date range
        filt_corp_act_df = ohlcv_df[
            (ohlcv_df['CORP_ACTION'].str.contains('Dividend', case=False, na=False)) &
            (ohlcv_df['DATE'].between(start_date, end_date))
        ][['DATE', 'CORP_ACTION']]

        # Initialize ADJ_CLOSE_DIV only for the specified date range
        mask = ohlcv_df['DATE'].between(start_date, end_date)
        ohlcv_df.loc[mask, 'ADJ_CLOSE_DIV'] = ohlcv_df.loc[mask, 'CLOSE']

        if filt_corp_act_df.empty:
            return ohlcv_df  # No dividends in the range, return early

        for _, row in filt_corp_act_df[::-1].iterrows():
            ex_date_dt = row['DATE']
            purposes = re.findall(r'<(.*?)\/>', row['CORP_ACTION'])  # Extract corporate actions
            
            for purpose in purposes:
                if re.search(r'\b(Dividend)\b', purpose) and purpose != 'Dividend-Nil':
                    mask = ohlcv_df['DATE'] < ex_date_dt
                    filt_df = ohlcv_df[mask]

                    if not filt_df.empty:
                        prev_close = filt_df.iloc[-1]['CLOSE']
                        action, factor = self.__get_action_factor(purpose, prev_close)
                        
                        if action is not None:
                            ohlcv_df.loc[mask, 'ADJ_CLOSE_DIV'] *= factor
                            ohlcv_df.loc[:, 'ADJ_CLOSE_DIV'] = ohlcv_df.loc[:, 'ADJ_CLOSE_DIV'].astype(float).round(2)

        return ohlcv_df


    def __update_stock_corporate_action_in_OHLCV(self, ohlcv_df:pd.DataFrame, corp_act_df:pd.DataFrame, start_date:datetime, end_date:datetime):
        # Ensure 'CORP_ACTION' column exists in ohlcv_df
        if 'CORP_ACTION' not in ohlcv_df.columns:
            ohlcv_df['CORP_ACTION'] = ''
        
        # Iterate over corporate actions and append them to the corresponding date in ohlcv_df
        for index, row in corp_act_df.iterrows():
            date = row['EX-DATE']
            purpose = row['PURPOSE']
            
            if start_date <= date and date <= end_date:
                if date in ohlcv_df['DATE'].values:
                    ohlcv_df.loc[ohlcv_df['DATE'] == date, 'CORP_ACTION'] = \
                        ohlcv_df.loc[ohlcv_df['DATE'] == date, 'CORP_ACTION'].astype(str) + '<' + purpose + '/>'
                else:
                    self.__logger.error(f'{datetime.strftime(date, "%d-%b-%Y")} not available in ohlcv dataframe')
        
        return ohlcv_df


    def __extractOHLCV(self, nse_df):
        try:
            columns = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
            OHLCV_df = nse_df[columns]
            OHLCV_df = OHLCV_df.rename(columns={'Date': 'DATE', 'close': 'CLOSE'})
            OHLCV_df.loc[:, ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']] = OHLCV_df[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']].replace({',': ''}, regex=True).astype(float)
            OHLCV_df.loc[:, ['VOLUME']] = OHLCV_df[['VOLUME']].replace({',': ''}, regex=True).infer_objects(copy=False)
        except Exception as e:
            OHLCV_df = None

        return OHLCV_df


    def __recurse_over_symbols(self, nse_ohlcv_df:pd.DataFrame, nse_corp_act_df:pd.DataFrame, 
                               adjust_for_split_bonus:bool, adjust_for_div:bool,
                               start_date:datetime, end_date:datetime):
        adjusted_dfs = []  # List to store results

        # Get unique symbols present in both dataframes
        unique_symbols = set(nse_corp_act_df["SYMBOL"]).intersection(nse_ohlcv_df["SYMBOL"].unique())

        # Process symbols that require adjustments
        for symbol in unique_symbols:
            ohlcv_subset = nse_ohlcv_df[nse_ohlcv_df["SYMBOL"] == symbol]
            ohlcv_subset['DATE'] = pd.to_datetime(ohlcv_subset['DATE'], format='%d-%b-%Y')
            corp_act_subset = nse_corp_act_df[nse_corp_act_df["SYMBOL"] == symbol]
            corp_act_subset['EX-DATE'] = pd.to_datetime(corp_act_subset['EX-DATE'], format='%d-%b-%Y')
            
            ohlcv_subset = self.__update_stock_corporate_action_in_OHLCV(ohlcv_subset, corp_act_subset, start_date, end_date)
            if adjust_for_split_bonus:
                ohlcv_subset = self.__adjust_for_split_bonus(ohlcv_subset, start_date, end_date)
            if adjust_for_div:
                ohlcv_subset = self.__adjust_for_div(ohlcv_subset, start_date, end_date)
            ohlcv_subset['DATE'] = pd.to_datetime(ohlcv_subset['DATE']).dt.strftime('%d-%b-%Y')
            adjusted_dfs.append(ohlcv_subset)

        # Append unmodified symbols (those not in nse_corp_act_df)
        unmodified_symbols = set(nse_ohlcv_df["SYMBOL"].unique()) - unique_symbols
        adjusted_dfs.append(nse_ohlcv_df[nse_ohlcv_df["SYMBOL"].isin(unmodified_symbols)])

        # Concatenate results
        return pd.concat(adjusted_dfs, ignore_index=True)
    
    
    def recurse_over_symbols_for_div_adj(self, nse_ohlcv_df:pd.DataFrame,  
                                         adjust_for_div:bool,
                                         start_date:datetime, end_date:datetime):
        adjusted_dfs = []  # List to store results

        # Get unique symbols present in both dataframes
        unique_symbols = set(nse_ohlcv_df['SYMBOL'])

        # Process symbols that require adjustments
        for symbol in unique_symbols:
            ohlcv_subset = nse_ohlcv_df[nse_ohlcv_df["SYMBOL"] == symbol]
            ohlcv_subset['DATE'] = pd.to_datetime(ohlcv_subset['DATE'], format='%d-%b-%Y')
            if adjust_for_div:
                ohlcv_subset['ADJ_CLOSE_DIV'] = ohlcv_subset['CLOSE']
                ohlcv_subset = self.__adjust_for_div(ohlcv_subset, start_date, end_date)
            ohlcv_subset['DATE'] = pd.to_datetime(ohlcv_subset['DATE']).dt.strftime('%d-%b-%Y')
            adjusted_dfs.append(ohlcv_subset)

        # Concatenate results
        return pd.concat(adjusted_dfs, ignore_index=True)


    def get_nse_index_constitutents(self, index:str):
        url = {"NIFTY750": "https://www.niftyindices.com/IndexConstituent/ind_niftytotalmarket_list.csv",
               "NIFTY500": "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
               "NIFTY50": "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv"}

        index_df = self.__download_csv(url[index], host='www.niftyindices.com')
        index_df = index_df.rename(columns={'Company Name': 'COMPANY_NAME', 'Industry': 'INDUSTRY', 'Symbol': 'SYMBOL', 'Series': 'SERIES', 'ISIN Code': 'ISIN_CODE'})
        index_df['SYMBOL'] = index
        return index_df


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
    

    def change_time_period(self, nse_ohlcv_df:pd.DataFrame, timeperiod:str) -> pd.DataFrame:
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
        timeperiod = timeperiod.upper()
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
            self.__logger.error(f"Invalid period type: {period_type}. Must be one of 'D', 'W', 'M', 'Q', 'Y'.")

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


    def get_stock_corporate_action_data(self, symbol):
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

        # Update request parameters
        params = {
            "index": "equities",
            "symbol": symbol,
            "csv": "true"
        }

        nse_df = self.__download_csv(url, host = 'www.nseindia.com', params=params)
        
        # Check if response is empty
        if nse_df is None:
            self.__logger.info("No more data to fetch.")
        else:
            self.__logger.info(f"Fetched coporate action data for {symbol}.")
            nse_df = nse_df.rename(columns={nse_df.columns[0]: "SYMBOL"})

        return nse_df 


    def get_all_stocks_corporate_action_data(self, start_date:str, end_date:str=None):
            """
            Retrieves all corporate actions data from the NSE website for a specified date range.

            Args:
                start_date (str): The start date of the date range in the format 'dd-MMM-yyyy'.
                end_date (str, optional): The end date of the date range in the format 'dd-MMM-yyyy'. Defaults to None, in 
                                          which case corporate action only for the start_date is fetched

            Returns:
                pandas.DataFrame: A DataFrame containing the corporate actions data.

            Raises:
                ValueError: If the start_date is not in the correct format.

            Example:
                nse_data = NSEData()
                start_date = '01-Jan-2022'
                end_date = '31-Jan-2022'
                corp_act_df = nse_data.get_all_corporate_actions(start_date, end_date)
                print(corp_act_df)
            """
            url = "https://www.nseindia.com/api/corporates-corporateActions?index=equities&from_date=<FROM_DATE>&to_date=<TO_DATE>&csv=true"
            start_date = datetime.strptime(start_date, '%d-%b-%Y').strftime('%d-%m-%Y')
            end_date = datetime.strptime(end_date, '%d-%b-%Y').strftime('%d-%m-%Y') if end_date is not None else start_date
            url = re.sub('<FROM_DATE>', start_date, url)
            url = re.sub('<TO_DATE>', end_date, url)
            corp_act_df = self.__download_csv(url, host='www.nseindia.com')
            corp_act_df = corp_act_df.rename(columns={corp_act_df.columns[0]: "SYMBOL"})
            corp_act_df = corp_act_df.rename(columns={'COMPANY NAME': 'COMPANY_NAME', 'RECORD DATE': 'RECORD_DATE', 'BOOK CLOSURE START DATE': 'BOOK_CLOSURE_START_DATE', 
                                                    'BOOK CLOSURE END DATE': 'BOOK_CLOSURE_END_DATE'})
            return corp_act_df


    def get_stock_OHLCV_data(self, symbol: str, start: str, end:str=None, 
                             adjust_for_split_bonus:bool=True, adjust_for_div:bool = True, timeperiod:str='1D') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns the historical data of a NSE traded stock as a pandas dataframe

        Args:
            symbol (str): [Required] NSE symbol for which the historical data needs to be downloaded
            start (str): [Required] Start date in the form of dd-mmm-yyyy indicating from when the historical data is needed. Example: '15-Sep-2008'
            end (str): [Optional] End date in the form of dd-mmm-yyyy indicating until when the data is needed. Example: '15-Sep-2008'. Default: Today's date
            adjust_for_split_bonus (bool): [Optional] A boolean variable indicating if the stock price should be adjusted for corporate actions (bonus and stock splits). Default = True
            adjust_for_div (bool): [Optional] A boolean variable indicating if the stock price should be adjusted for dividends. If set to True, will assume adjust_for_split_bonus is also True. Default = True
            timeperiod (str) : [Optional] Aggregate stock quote so that every row in the dataframe corresponds to this duration - 1W, 2W, 1M, 1Q, 1Y. Default = '1D'

        Returns:
            pd.DataFrame: 
            - The dataframe contains the following columns - 'DATE' 'OPEN' 'HIGH' 'LOW' 'CLOSE' 'VOLUME' and 'SYMBOL'. The dataframe can be indexed using the 'DATE' column.
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
            end_obj = datetime.strptime(end, '%d-%b-%Y')
            start_obj = datetime.strptime(start, '%d-%b-%Y')
            nse_df = self.__download_historical_price_volume(symbol, series, start_obj, end_obj)
            nse_ohlcv_df = self.__extractOHLCV(nse_df)
            
            if nse_ohlcv_df is not None:
                if adjust_for_split_bonus or adjust_for_div:
                    nse_ohlcv_df['CORP_ACTION'] = ''
                    nse_corp_act_df = self.get_stock_corporate_action_data(symbol)
                    # Sort DataFrame by date in ascending order, because that's required by libta
                    nse_ohlcv_df.loc[:, 'DATE'] = pd.to_datetime(nse_ohlcv_df['DATE'], format='%d-%b-%Y')
                    nse_ohlcv_df = nse_ohlcv_df.sort_values(by='DATE')
                    nse_corp_act_df.loc[:, 'EX-DATE'] = pd.to_datetime(nse_corp_act_df['EX-DATE'], format='%d-%b-%Y', errors='coerce')
                    # Copy the corporate action entries from nse_corp_act_df into nse_ohlcv_df
                    nse_ohlcv_df = self.__update_stock_corporate_action_in_OHLCV(nse_ohlcv_df, nse_corp_act_df, start_obj, end_obj)

                    if adjust_for_div:
                        adjust_for_split_bonus = True
                        
                    if adjust_for_split_bonus:
                        nse_ohlcv_df = self.__adjust_for_split_bonus(nse_ohlcv_df, start_obj, end_obj)
                    if adjust_for_div:
                        nse_ohlcv_df = self.__adjust_for_div(nse_ohlcv_df, start_obj, end_obj)

                nse_ohlcv_df['DATE'] = pd.to_datetime(nse_ohlcv_df['DATE']).dt.strftime('%d-%b-%Y')
        nse_ohlcv_df['SYMBOL'] = symbol
        return nse_ohlcv_df


    def update_stock_OHLCV_from_bhav(self, nse_ohlcv_df:pd.DataFrame, start_date:str, end_date:str=None, 
                                     adjust_for_split_bonus:bool=True, adjust_for_div:bool = True, special_trading_day=[], holiday_list=[]):
        """
        Updates the stock OHLCV (Open, High, Low, Close, Volume) data from the NSE Bhavcopy for a given date range.

        Args:
            nse_ohlcv_df (pd.DataFrame): The existing OHLCV dataframe to be updated.
            start_date (str): The start date of the date range in the format "%d-%b-%Y".
            end_date (str, optional): The end date of the date range in the format "%d-%b-%Y". Defaults to None
            special_trading_day (list, optional): List of special trading days. Defaults to an empty list.
            holiday_list (list, optional): List of holidays. Defaults to an empty list.

        Returns:
            pd.DataFrame: The updated OHLCV dataframe.

        Raises:
            None

        Example:
            # Create an empty OHLCV dataframe
            nse_ohlcv_df = pd.DataFrame(columns=["SYMBOL", "DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"])

            # Update the OHLCV dataframe from the NSE Bhavcopy for a specific date range
            updated_df = update_stock_OHLCV_from_bhav(nse_ohlcv_df, "01-Jan-2022", "31-Jan-2022")
        """
        nse_ohlcv_df["DATE"] = pd.to_datetime(nse_ohlcv_df["DATE"], format="%d-%b-%Y")
        
        # Convert start_date to datetime
        start_date_obj = datetime.strptime(start_date, "%d-%b-%Y")
        # Get current date
        end_date_obj = datetime.strptime(end_date, "%d-%b-%Y") if end_date is not None else start_date_obj
        # Convert holiday_list to datetime for easy comparison
        special_trading_day = set(pd.to_datetime(special_trading_day, format="%d-%b-%Y").date)
        holiday_list = set(pd.to_datetime(holiday_list, format="%d-%b-%Y").date)

        # Create set of SYMBOLs in the df
        symbols = set(nse_ohlcv_df["SYMBOL"])  # Convert SYMBOL column to a set for faster lookup

        # Iterate over days, skipping weekends and holidays
        current_date = start_date_obj
        while current_date <= end_date_obj:
            if (current_date.date() in special_trading_day and current_date.weekday() >= 5) or (current_date.date() not in holiday_list and current_date.weekday() < 5):
                bhav_df = self.get_bhav(datetime.strftime(current_date, "%d-%b-%Y"))
                if bhav_df is not None and not bhav_df.empty:
                    bhav_df = bhav_df[bhav_df["SYMBOL"].isin(symbols)]  # Filter only symbols in index_df
                    # Ensure DATE column is in datetime format
                    bhav_df["DATE"] = pd.to_datetime(bhav_df["DATE"], format="%d-%b-%Y")

                    # Identify rows that do not already exist in stock_TI_df
                    filt_nse_ohlcv_df = nse_ohlcv_df[nse_ohlcv_df["DATE"] == current_date]
                    existing_rows = set(filt_nse_ohlcv_df["SYMBOL"])
                    new_rows = bhav_df[~bhav_df.set_index(["SYMBOL"]).index.isin(existing_rows)]

                    if not new_rows.empty:
                        if adjust_for_split_bonus or adjust_for_div:
                            new_rows['CORP_ACTION'] = ''
                        nse_ohlcv_df = pd.concat([nse_ohlcv_df, new_rows], ignore_index=True)
            current_date += timedelta(days=1)  # Move to next day

        nse_ohlcv_df['DATE'] = pd.to_datetime(nse_ohlcv_df['DATE']).dt.strftime('%d-%b-%Y')

        nse_corp_act_df = self.get_all_stocks_corporate_action_data(start_date, end_date)
        if not nse_corp_act_df.empty:
            nse_corp_act_df = nse_corp_act_df[nse_corp_act_df['SYMBOL'].isin(symbols)]
            if not nse_corp_act_df.empty:
                if adjust_for_split_bonus or adjust_for_div:
                    if adjust_for_div:
                        adjust_for_split_bonus = True
                    nse_ohlcv_df = self.__recurse_over_symbols(nse_ohlcv_df, nse_corp_act_df, 
                                                               adjust_for_split_bonus, adjust_for_div, 
                                                               start_date_obj, end_date_obj)

        return nse_ohlcv_df


    def get_bhav(self, date:str):
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
                bhav_df = None
                # Send an initial request to the main site to get cookies and headers
                session.get('https://www.nseindia.com', headers=self.headers)   

                # Send a GET request
                self.headers['Host'] = 'nsearchives.nseindia.com'
                response = session.get(url, headers=self.headers)
                response.raise_for_status()  # Raise exception for HTTP errors
                
                # Return the content of the response
                self.__logger.info(f"File downloaded successfully from URL: {url}")
                
                # Extract the ZIP file contents
                with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                    # Assume the ZIP contains exactly one CSV file
                    csv_filename = zf.namelist()[0]
                    self.__logger.info(f"Extracting file: {csv_filename}")
                    
                    # Read the CSV file into a DataFrame
                    with zf.open(csv_filename) as csv_file:
                        bhav_df = pd.read_csv(csv_file)
                bhav_df = bhav_df[bhav_df['SctySrs'].isin(['EQ', 'BE', 'BZ', 'SM', 'ST', 'SZ'])]
                bhav_df = bhav_df[['TradDt', 'TckrSymb', 'OpnPric', 'HghPric', 'LwPric', 'ClsPric', 'TtlTradgVol']]
                bhav_df.columns = ['DATE', 'SYMBOL', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
                bhav_df['DATE'] = pd.to_datetime(bhav_df['DATE'], format="%Y-%m-%d").dt.strftime('%d-%b-%Y')
                self.__logger.info(f"File downloaded and read successfully from URL: {url}")
            except requests.exceptions.RequestException as e:
                self.__logger.error(f"Error downloading file: {e}")
            except zipfile.BadZipFile as e:
                self.__logger.error(f"Error extracting ZIP file: {e}")
            except Exception as e:
                self.__logger.error(f"Unexpected error: {e}")
            return bhav_df


    def get_index_bhav(self, date:str):
        url = "https://www.niftyindices.com/Daily_Snapshot/ind_close_all_<DATE>.csv"
        date = datetime.strptime(date, '%d-%b-%Y').strftime('%d%m%Y')
        url = re.sub('<DATE>', date, url)
        bhav_df = self.__download_csv(url, host='www.niftyindices.com')
        if not bhav_df.empty:
            bhav_df = bhav_df[['Index Name', 'Index Date', 'Open Index Value', 'High Index Value', 'Low Index Value', 'Closing Index Value', 'Volume', 'P/E', 'P/B', 'Div Yield']]
            bhav_df.columns = ['INDEX_NAME', 'DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'P/E', 'P/B', 'DIV_YIELD']
            bhav_df['DATE'] = pd.to_datetime(bhav_df['DATE'], format="%d-%m-%Y").dt.strftime('%d-%b-%Y')
        return bhav_df


    