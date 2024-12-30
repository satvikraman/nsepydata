import os
import pandas as pd
import sys

sys.path.append('./src')
import nsepydata

histApp = nsepydata.NSEPyData()

folder = './temp/'
symbol = 'NESTLEIND'

os.makedirs(folder, exist_ok=True)

nse_df, nse_corp_act_df = histApp.download_historical_data(symbol, 'EQ', '01-Jan-2023', '12-Dec-2024', True)
nse_df.to_csv(folder+symbol+'_raw.csv', index=False)
nse_corp_act_df.to_csv(folder+symbol+'_corp_action.csv', index=False)

nse_df = pd.read_csv(folder+symbol+'_raw.csv')
nse_corp_act_df = pd.read_csv(folder+symbol+'_corp_action.csv')
nse_df = histApp.getOHLCV(nse_df)
nse_df = histApp.adjust_for_corp_action(nse_df, nse_corp_act_df)
nse_df.to_csv(folder+symbol+'.csv', index=False)