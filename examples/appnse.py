import os
import pandas as pd
import sys

sys.path.append('./src')
from nsepydata import NSEPyData

obj = NSEPyData()

folder = './temp/'
symbol = 'NMSTEEL'

os.makedirs(folder, exist_ok=True)

#nse_df = obj.get_OHLCV_data(symbol='NESTLEIND', start='01-Dec-2024', end='31-Dec-2024', adjust_corp_action=True, timeperiod='1D')
#nse_df.to_csv(folder+symbol+'_1D.csv', index=False)
#nse_corp_act_df = histApp.get_corporate_action_data(symbol)
#nse_corp_act_df.to_csv(folder+symbol+'_corpact.csv', index=False)

ohlcv_data = obj.get_OHLCV_data(symbol='NESTLEIND', start='01-Dec-2023', end='31-Dec-2024', adjust_corp_action=True, timeperiod='1D')
nse_df.to_csv(folder+symbol+'_1D.csv', index=False)
