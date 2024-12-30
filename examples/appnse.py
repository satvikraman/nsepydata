import os
import pandas as pd
import sys

sys.path.append('./src')
import nsepydata

histApp = nsepydata.NSEPyData()

folder = './temp/'
symbol = 'NESTLEIND'

os.makedirs(folder, exist_ok=True)

nse_df, nse_corp_act_df = histApp.get_OHLCV_data(symbol=symbol, series='EQ', start='01-Feb-2023', end='12-Dec-2024', adjust_corp_action=True, timeperiod='2M')
nse_df.to_csv(folder+symbol+'_2M.csv', index=False)