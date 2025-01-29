import os
import pandas as pd
import sys

sys.path.append('./src')
from nsepydata import NSEPyData

obj = NSEPyData()

folder = './temp/'

os.makedirs(folder, exist_ok=True)

symbol = 'NESTLEIND'
nse_df = obj.get_OHLCV_data(symbol='NESTLEIND', start='01-Jan-2024', end='29-Jan-2025', adjust_for_split_bonus=True, adjust_for_div=True, timeperiod='1D')
nse_df.to_csv(folder+symbol+'_1D.csv', index=False)

#symbol = 'NESTLEIND'
#nse_corp_act_df = obj.get_corporate_action_data(symbol)
#nse_corp_act_df.to_csv(folder+symbol+'_corpact.csv', index=False)

#bhav_df = obj.download_bhav('27-Jan-2025')
#bhav_df.to_csv(folder+'nsebhav.csv', index=False)