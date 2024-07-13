from datetime import datetime, timedelta, timezone
from scripts.find_dumps import convert_unix_to_utc_plus_3
from scripts.aggregate_data import get_dfs_aggregated
from scripts.find_dumps import find_dump_intervals

# get aggregated data for dumps intervals from 1s periods
def get_last_data(minutes_to_see,period, output_path=None):
    if period=='15s':
        last_rows_to_see=minutes_to_see*60/15
    else:
        raise ValueError('rewrite get_last_data')
    
    current_time = datetime.now(timezone.utc)
    # Round down to the previous 15 seconds
    seconds = (current_time - current_time.replace(second=0, microsecond=0)).seconds
    rounding = (seconds // 15) * 15
    current_time = current_time.replace(second=0, microsecond=0) + timedelta(seconds=rounding)
    current_time = int(current_time.timestamp() * 1000)
    end_time=int(current_time-15000*(last_rows_to_see+1))
    
    # more rows, since last row is current, and have not complete data, hence useless. First row has invalid values (.diff calculation)
    timestamps_unix = [(end_time,current_time)]
    
    print("current period")
    for timestamp_unix in timestamps_unix:
        print(convert_unix_to_utc_plus_3(timestamp_unix[0]),convert_unix_to_utc_plus_3(timestamp_unix[1]))
    
    dfs_aggregated=get_dfs_aggregated(timestamps_unix, agg_trades_source='klines')
    # for df in dfs_aggregated:
    #     for column in df.columns:
    #         if column!='unix':
    #             print(column, max(df[column]),min(df[column]))
    
    if output_path==None:
        return dfs_aggregated[0]
    else:
        dfs_aggregated[0].to_csv(output_path+".csv", index=False)

def get_dumps_data(output_path=None):
    # get aggregated data for dumps intervals from 1s periods
    timestamps_unix = find_dump_intervals()
    
    print("dumps period")
    for timestamp_unix in timestamps_unix:
        print(convert_unix_to_utc_plus_3(timestamp_unix[0]),convert_unix_to_utc_plus_3(timestamp_unix[1]))
    
    # add 15 additional min after and before dump, to give model more context
    timestamps_unix_adjusted=[]
    for start_unix, end_unix in timestamps_unix:
        timestamps_unix_adjusted.append((start_unix-900_000,end_unix+900_000))
    
    dfs_aggregated=get_dfs_aggregated(timestamps_unix_adjusted)
    # for df in dfs_aggregated:
    #     for column in df.columns:
    #         if column!='unix':
    #             print(column, max(df[column]),min(df[column]))
    
    if output_path==None:
        return dfs_aggregated
    else:
        for i in range (len(dfs_aggregated)):
            dfs_aggregated[i].to_csv(output_path+str(i)+".csv", index=False)
