import os
import pandas as pd
from get_binance_data import get_klines_futures, get_full_klines_spot

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

def aggregate_for_timeframe(df, name_aggregation_column, amount_seconds):    
    assert 'unix' in df.columns, "DataFrame must contain 'unix' column"
    assert name_aggregation_column in df.columns, f"DataFrame must contain '{name_aggregation_column}' column"

    period_ms = period * 1000
    df['time_bucket'] = (df['unix'] // period_ms) * period_ms
    
    aggregated_df = df.groupby('time_bucket')[name_aggregation_column].sum().reset_index()
    aggregated_df.rename(columns={'time_bucket': 'unix', name_aggregation_column: 'agg'+name_aggregation_column}, inplace=True)
    
    return aggregated_df

def create_bin(df, bin_column_name, period):
    assert bin_column_name in df.columns, f"DataFrame must contain '{bin_column_name}' column"
    
    # Convert the bin_column_name to numeric, coercing errors to NaN
    df[bin_column_name] = pd.to_numeric(df[bin_column_name], errors='coerce')
    
    # Drop rows with NaN values in bin_column_name
    df = df.dropna(subset=[bin_column_name])
    
    # Ensure the column is of integer type
    df[bin_column_name] = df[bin_column_name].astype(int)
    
    # Calculate the bin column
    df['bin'] = (df[bin_column_name] // (period * 1000)) * (period * 1000)
    
    # Sort the DataFrame by the bin column
    df = df.sort_values('bin')
    
    return df

symbol='BTCUSDT'
period='15s'
directory="aggTrades"
dumps_unix = [(1716471900000, 1716473100000),
(1717426200000, 1717426500000),
(1717782600000, 1717784100000),
(1718070600000, 1718070900000),
(1718220000000, 1718221800000),
(1718380500000, 1718381400000)]

files=get_all_file_names(directory)
for dump_unix in dumps_unix:
    # for specified dump period, aggregate all data
    # aggTrades
    for file in files:
        file_path = os.path.join(directory, file)
        df_aggtrades=pd.read_csv(file_path)
        df_aggtrades.columns=['id', 'price','quantity','firstid','lastid','unix','takersell','pricechanged']
        
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

    # create corresponding bin
    # TODO: fix
    # df_aggtrades_dump=create_bin(df_aggtrades_dump,'unix',period)
    # print(df_aggtrades_dump.iloc[0])
    
    # spot delta change%, volume change%, price change% by second
    klines_spot=get_full_klines_spot(symbol,'1s',dump_unix[0],dump_unix[1])
    df_klines = pd.DataFrame(klines_spot, columns=['unix', 'Open', 'High', 'Low', 'Close', 'Volume', 
    'Close time', 'Quote asset volume', 'Number of trades', 
    'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
    
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume', 'Taker buy base asset volume', 'Taker buy quote asset volume']
    df_klines[numeric_cols] = df_klines[numeric_cols].apply(pd.to_numeric)
        
    df_klines['Taker sell base asset volume'] = df_klines['Volume']-df_klines['Taker buy base asset volume']
    df_klines['Taker volume delta']=df_klines['Taker buy base asset volume']-df_klines['Taker sell base asset volume']
    
    df_open=aggregate_for_timeframe(df_klines,'Open',period)
    df_volume=aggregate_for_timeframe(df_klines,'Volume',period)
    df_volume_delta=aggregate_for_timeframe(df_klines,'Taker volume delta',period)
    
    df_open_change=df_open['agg Open'].diff()
    df_volume_change=df_volume['agg Volume'].diff()
    df_volume_delta_change=df_volume_delta['agg Taker volume delta'].diff()

# draw plots for time series - ty
# automate - We
# 1,2 - Ya