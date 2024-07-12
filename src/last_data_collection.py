from datetime import datetime, timedelta, timezone
from scripts.find_dumps import convert_unix_to_utc_plus_3
from scripts.aggregate_data import get_dfs_aggregated
# get aggregated data for dumps intervals from 1s periods
if __name__=="__main__":
    # get last 20 minutes data
    last_rows_to_see=80
    output_path=f"data/raw/last_20_minutes_data"
    
    current_time = datetime.now(timezone.utc)
    # Round down to the previous 15 seconds
    seconds = (current_time - current_time.replace(second=0, microsecond=0)).seconds
    rounding = (seconds // 15) * 15
    current_time = current_time.replace(second=0, microsecond=0) + timedelta(seconds=rounding)
    current_time = int(current_time.timestamp() * 1000)
    
    # more rows, since last row is current, and have not complete data, hence useless. First row has invalid values (.diff calculation)
    timestamps_unix = [(current_time-15000*(last_rows_to_see+1),current_time)]
    
    print("current period")
    for timestamp_unix in timestamps_unix:
        print(convert_unix_to_utc_plus_3(timestamp_unix[0]),convert_unix_to_utc_plus_3(timestamp_unix[1]))
    
    dfs_aggregated=get_dfs_aggregated(timestamps_unix, agg_trades_source='klines')
    for df in dfs_aggregated:
        for column in df.columns:
            if column!='unix':
                print(column, max(df[column]),min(df[column]))
    
    dfs_aggregated[0].to_csv(output_path+".csv", index=False)

# TODO: external/aggTrades dvc
# run 'python -m src.last_data_collection', runtime = 1.02s