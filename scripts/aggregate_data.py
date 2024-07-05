import sys
import os
from datetime import datetime, timedelta, timezone
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

import pandas as pd
from functools import reduce
from src.data.scraping.get_historical_binance_data import get_full_klines_spot
from scripts.find_dumps import find_dump_intervals
from scripts.find_dumps import convert_unix_to_utc_plus_3

def get_all_file_names(directory):
    all_files = os.listdir(directory)
    files = [f for f in all_files if os.path.isfile(os.path.join(directory, f))]
    return files

def get_last_month_dates():
    now = datetime.now(timezone.utc)
    
    end_date = now - timedelta(days=1)
    start_date = end_date - timedelta(days=29)
    
    unix_timestamps = []
    current_day = start_date
    while current_day <= end_date:
        start_of_day = datetime(current_day.year, current_day.month, current_day.day, tzinfo=timezone.utc)
        end_of_day = start_of_day + timedelta(days=1) - timedelta(seconds=1)
        
        # Convert to Unix timestamp (milliseconds)
        start_of_day_unix = int(start_of_day.timestamp() * 1000)
        end_of_day_unix = int(end_of_day.timestamp() * 1000)
        
        unix_timestamps.append((start_of_day_unix, end_of_day_unix))
        current_day += timedelta(days=1)
    
    return unix_timestamps

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
    
    if name_aggregation_column=="Open" or name_aggregation_column=="Low" or name_aggregation_column=="Close":
        aggregated_df['agg_'+name_aggregation_column]/=(period_ms/1000)
    
    return aggregated_df[['unix', 'agg_' + name_aggregation_column]]

def preprocess_aggregated_df(df):
    able_to_preprocess_columns=['agg_Close','agg_Low','agg_Volume']
    if 'agg_Close' in df.columns:
        df['Close']=df['agg_Close']
    
    for column in able_to_preprocess_columns:
        if column in df.columns:
            df[column+'_diff']=df[column].diff().fillna(0)
            df=df.drop(column,axis=1)
            
    if 'trades' in df.columns:
        df['amount_trades'] = df['trades'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df=df.drop('trades', axis=1)
    return df

def create_bin(df, bin_column_name, period):
    assert bin_column_name in df.columns, f"DataFrame must contain '{bin_column_name}' column"
    
    if period == '15s':
        period_ms = 15 * 1000
    elif period == '60s':
        period_ms = 60 * 1000
    else:
        raise ValueError("Period must be '15s' or '60s'")
    
    df = df.copy()
    df.loc[:, 'unix'] = (df['unix'] // period_ms) * period_ms
    return df

def merge_dfs(left, right):
    return pd.merge(left, right, on='unix', how='outer')

def get_dfs_aggregated(periods_unix, symbol='BTCUSDT',period='15s'):
    script_dir = os.path.dirname(__file__)
    directory=os.path.join(script_dir, '..', 'data', 'external', 'aggtrades')
   
    files=get_all_file_names(directory)
    print("training periods:")
    for period_unix in periods_unix:
        print(convert_unix_to_utc_plus_3(period_unix[0]),convert_unix_to_utc_plus_3(period_unix[1]))
    
    dfs_dump_aggregated=[]
    for period_unix in periods_unix:
        # for specified period, aggregate all data
        
        # aggTrades
        # taken from binance website. If use api, too long to wait. Files contain aggtrades for specific day
        df_aggtrades = pd.DataFrame()
        aggTrades_file_found=False
        # for every learning period, find .csv file with trades
        for file in files:
            file_path = os.path.join(directory, file)
            df_aggtrades=pd.read_csv(file_path)
            df_aggtrades.columns=['id', 'price','quantity','firstid','lastid','unix','takersell','pricechanged']
            df_aggtrades=df_aggtrades[['price','quantity','unix','takersell','pricechanged']]
            
            start_day_unix=df_aggtrades['unix'].iat[0]
            end_day_unix=df_aggtrades['unix'].iat[-1]
            
            if start_day_unix//1000<=period_unix[0]//1000<=end_day_unix//1000 and start_day_unix//1000<=period_unix[1]//1000<=end_day_unix//1000:
                # corresponding dump period found
                df_aggtrades=df_aggtrades[(df_aggtrades['unix'] >= period_unix[0]) & (df_aggtrades['unix'] <= period_unix[1])]
                # print(len(df_aggtrades_dump)/len(df_aggtrades))
                aggTrades_file_found=True
                break
            
        if not aggTrades_file_found:
            print ("No data found for some interval. Download data from https://www.binance.com/en-NG/landing/data")
            continue
        df_aggtrades=create_bin(df_aggtrades,'unix',period)
        
        # spot delta change%, volume change%, price change% by second
        klines_spot=get_full_klines_spot(symbol,'1s',period_unix[0],period_unix[1])
        df_klines = pd.DataFrame(klines_spot, columns=['unix', 'Open', 'High', 'Low', 'Close', 'Volume', 
        'Close time', 'Quote asset volume', 'Number of trades', 
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
        
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume', 'Taker buy base asset volume', 'Taker buy quote asset volume']
        df_klines[numeric_cols] = df_klines[numeric_cols].apply(pd.to_numeric)
            
        df_klines['Taker sell base asset volume'] = df_klines['Volume']-df_klines['Taker buy base asset volume']
        df_klines['Taker volume delta']=df_klines['Taker buy base asset volume']-df_klines['Taker sell base asset volume']
        
        # all available data
        df_close=aggregate_for_timeframe(df_klines,'Close',period)
        df_volume=aggregate_for_timeframe(df_klines,'Volume',period)
        df_volume_delta=aggregate_for_timeframe(df_klines,'Taker volume delta',period)
        df_aggtrades
        grouped = df_aggtrades.groupby('unix')
        grouped_trades = []
        for unix, group in grouped:
            grouped_trades.append({'unix': unix, 'records': group.drop(columns='unix').to_dict('records')})
        df_grouped_trades = pd.DataFrame(grouped_trades)
        df_grouped_trades.columns = ['unix', 'trades']
        
        # Concatenate dataframes
        dfs=[df_close,df_volume,df_volume_delta,df_grouped_trades]
        df_dump_aggregated = reduce(merge_dfs, dfs)
        
        df_dump_aggregated=preprocess_aggregated_df(df_dump_aggregated)
        
        dfs_dump_aggregated.append(df_dump_aggregated)
        
    return dfs_dump_aggregated

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

# get aggregated data for dumps intervals from 1s periods
if __name__=="__main__":
    output_path="data/raw/dumps_aggregated"
    timestamps_unix = find_dump_intervals()
    
    print("dumps period")
    for timestamp_unix in timestamps_unix:
        print(convert_unix_to_utc_plus_3(timestamp_unix[0]),convert_unix_to_utc_plus_3(timestamp_unix[1]))
    
    # add 15 additional min after and before dump, to let model understand when to sell
    timestamps_unix_adjusted=[]
    for start_unix, end_unix in timestamps_unix:
        timestamps_unix_adjusted.append((start_unix-900_000,end_unix+900_000))
    
    dfs_aggregated=get_dfs_aggregated(timestamps_unix_adjusted)
    for df in dfs_aggregated:
        for column in df.columns:
            if column!='unix':
                print(column, max(df[column]),min(df[column]))
    
    for i in range (len(dfs_aggregated)):
        dfs_aggregated[i].to_csv(output_path+str(i)+".csv", index=False)

# TODO: 4July data download
# TODO: external/aggTrades dvc, delete dvc
# TODO: first and last rows deleted manually, since have invalid values