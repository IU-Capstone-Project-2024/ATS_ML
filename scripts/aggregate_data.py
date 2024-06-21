import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

import pandas as pd
from functools import reduce
from src.data.scraping.get_historical_binance_data import get_full_klines_spot
from scripts.find_dumps import find_dump_intervals

# historical 1s:
# spot trades
# price change% spot
# Volume change% spot

# historical 5min:
# Amount trades change% futures
# Liqudation amount change% futures
# Oi change% futures

# realtime:
# everything is available. Save data to local, for further improvement.
# Stakan plotnost bool spot+futures, round numbers bool

def get_all_file_names(directory):
    all_files = os.listdir(directory)
    files = [f for f in all_files if os.path.isfile(os.path.join(directory, f))]
    return files

def aggregate_for_timeframe(df, name_aggregation_column, period):
    # sum the values from period
    assert 'unix' in df.columns, "DataFrame must contain 'unix' column"
    assert name_aggregation_column in df.columns, f"DataFrame must contain '{name_aggregation_column}' column"

    if period == '15s':
        period_ms = 15 * 1000
    elif period == '60s':
        period_ms = 60 * 1000
    else:
        raise ValueError("Period must be '15s' or '60s'")

    df['time_bucket'] = (df['unix'] // period_ms) * period_ms
    aggregated_df = df.groupby('time_bucket')[name_aggregation_column].sum().reset_index()
    aggregated_df.rename(columns={'time_bucket': 'unix', name_aggregation_column: 'agg_' + name_aggregation_column}, inplace=True)
    
    if name_aggregation_column=="Open" or name_aggregation_column=="Low":
        aggregated_df['agg_'+name_aggregation_column]/=(period_ms/1000)
    
    return aggregated_df[['unix', 'agg_' + name_aggregation_column]]

def aggregate_trades(df_aggtrades):
    assert ''

def create_bin(df, bin_column_name, period):
    assert bin_column_name in df.columns, f"DataFrame must contain '{bin_column_name}' column"
    
    if period == '15s':
        period_ms = 15 * 1000
    elif period == '60s':
        period_ms = 60 * 1000
    else:
        raise ValueError("Period must be '15s' or '60s'")
    
    df['unix'] = (df['unix'] // period_ms) * period_ms
    
    return df

def merge_dfs(left, right):
    return pd.merge(left, right, on='unix', how='outer')

def dumps_aggregated(symbol='BTCUSDT',period='15s'):
    dumps_unix = find_dump_intervals()
    # TODO: save intervals in file
    
    script_dir = os.path.dirname(__file__)
    directory=os.path.join(script_dir, '..', 'data', 'external', 'aggtrades')
    files=get_all_file_names(directory)

    dumps_concatenated_df=[]
    for dump_unix in dumps_unix:
        
        # for specified dump period, aggregate all data
        # aggTrades
        # taken from binance website. If use api, too long to wait. Files contain aggtrades for specific day
        for file in files:
            file_path = os.path.join(directory, file)
            df_aggtrades=pd.read_csv(file_path)
            df_aggtrades.columns=['id', 'price','quantity','firstid','lastid','unix','takersell','pricechanged']
            df_aggtrades=df_aggtrades[['price','quantity','unix','takersell','pricechanged']]
            
            start_day_unix=df_aggtrades['unix'].iat[0]
            end_day_unix=df_aggtrades['unix'].iat[-1]
            
            df_aggtrades_dump = pd.DataFrame()
            if start_day_unix<=dump_unix[0]<=end_day_unix and start_day_unix<=dump_unix[1]<=end_day_unix:
                # corresponding dump period found
                df_aggtrades_dump=df_aggtrades[(df_aggtrades['unix'] >= dump_unix[0]) & (df_aggtrades['unix'] <= dump_unix[1])]
                # print(len(df_aggtrades_dump)/len(df_aggtrades))
                break
        if df_aggtrades_dump.empty:
            continue
        df_aggtrades_dump=create_bin(df_aggtrades_dump,'unix',period)
        
        # spot delta change%, volume change%, price change% by second
        klines_spot=get_full_klines_spot(symbol,'1s',dump_unix[0],dump_unix[1])
        df_klines = pd.DataFrame(klines_spot, columns=['unix', 'Open', 'High', 'Low', 'Close', 'Volume', 
        'Close time', 'Quote asset volume', 'Number of trades', 
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
        
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume', 'Taker buy base asset volume', 'Taker buy quote asset volume']
        df_klines[numeric_cols] = df_klines[numeric_cols].apply(pd.to_numeric)
            
        df_klines['Taker sell base asset volume'] = df_klines['Volume']-df_klines['Taker buy base asset volume']
        df_klines['Taker volume delta']=df_klines['Taker buy base asset volume']-df_klines['Taker sell base asset volume']
        
        # all available data
        df_low=aggregate_for_timeframe(df_klines,'Low',period)
        df_volume=aggregate_for_timeframe(df_klines,'Volume',period)
        df_volume_delta=aggregate_for_timeframe(df_klines,'Taker volume delta',period)
        df_aggtrades_dump
        
        df_grouped_trades = df_aggtrades_dump.groupby('unix').apply(lambda x: x.drop(columns='unix').to_dict('records')).reset_index()
        df_grouped_trades.columns=['unix','trades']
        
        # Concatenate dataframes
        dfs=[df_low,df_volume,df_volume_delta,df_grouped_trades]
        concatenated_df = reduce(merge_dfs, dfs)
        
        dumps_concatenated_df.append(concatenated_df)
        
    return dumps_concatenated_df

    # TODO: external/aggTrades dvc
    # TODO: label data
    # TODO: preprocessing
    # TODO: visualisations
    # TODO: automate everything