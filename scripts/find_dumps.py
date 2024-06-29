import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from src.data.scraping.get_historical_binance_data import get_klines_futures
from src.data.scraping.get_historical_binance_data import get_oi
from datetime import datetime, timezone, timedelta

def convert_unix_to_utc_plus_3(unix_timestamp):
    '''Convert Unix timestamp to datetime object in UTC'''
    
    dt_utc = datetime.fromtimestamp(int(unix_timestamp)/1000, tz=timezone.utc)
    utc_plus_3 = timezone(timedelta(hours=3))
    dt_utc_plus_3 = dt_utc.astimezone(utc_plus_3)
    return dt_utc_plus_3

def convert_index_to_unix(start_unix, period,index):
    if period=='1s':
        unix_period=1000
    elif period=='1m':
        unix_period=60000
    elif period=="5m":
        unix_period=300000
    elif period=="15m":
        unix_period=900000
    return start_unix+unix_period*index

def find_downfall_intervals_with_timestamps(values, cumulative_threshold, change_threshold, tolerance):
    """detects wide downfall periods from list of values.
        
        tolerance - how many growth values in row are considered as bad. Big tolerance partially fixed with cutting. 
        cumulative_threshold - downfall of whole interval
        change_treshold - treshold of value change to be accepted as downfall"""
    change_intervals = []  # list of tuples
    start_index = 1

    while start_index < len(values)-tolerance:
        cumulative_change = 0
        growth_bar = 0 # amount current growth in row bars
        # indexes of extremums are used to get accurate interval of downfall
        min_value_index = start_index -1
        max_value_index = start_index -1
        
        # start intervals only with change <=threshold
        if (float(values[start_index]) - float(values[start_index-1])) / float(values[start_index-1])>change_threshold:
            start_index+=1
            continue
        
        for iterator_index in range(start_index+1, len(values)):
            value_change_percent=(float(values[iterator_index]) - float(values[iterator_index-1])) / float(values[iterator_index-1])
            if float(values[iterator_index])>float(values[max_value_index]):
                max_value_index=iterator_index
            if float(values[iterator_index])<float(values[min_value_index]):
                min_value_index=iterator_index
            cumulative_change=(float(values[iterator_index]) - float(values[start_index])) / float(values[start_index])
            
            # if value downfall
            if value_change_percent < change_threshold:
                growth_bar=max(0,growth_bar-2)
            # penalty for growth value
            elif value_change_percent>0:
                growth_bar+=1
            # extra penalty for big growth
            elif value_change_percent>change_threshold:
                growth_bar+=1
            # little penalty for little growth
            elif value_change_percent>0:
                growth_bar+=0.25
            
            # if too much growth or dump is going right now
            if growth_bar>=tolerance or iterator_index==(len(values)-1):
                if ((float(values[max_value_index])-float(values[min_value_index]))/float(values[min_value_index])) > cumulative_threshold:
                    # cut edge decline/stagnate bars. Take precise growth period
                    change_intervals.append((cumulative_change, max_value_index, min_value_index))

                # end propogation with this start_index
                break
               
        # move start further
        start_index=min_value_index+2
    
    return change_intervals

def find_dumps(symbol, period, price_index, change_threshold, cumulative_threshold, tolerance, start_time=None, end_time=None, limit=None):
    # price index=2=lows. Work good for finding decresaing trend
    # get data from binance
    if limit:
        oi_data = get_oi(symbol, period, limit)
        klines_data=get_klines_futures(symbol,period,limit)
    if start_time and end_time:
        oi_data = get_oi(symbol, period, start_time=start_time,end_time=end_time)
        klines_data=get_klines_futures(symbol,period,start_time=start_time,end_time=end_time)
    
    timestamps=[] # list of unix timestamps for given period
    timestamps = [entry[0] for entry in klines_data]
    
    oi_values = [entry['sumOpenInterest'] for entry in oi_data]
    oi_downfall_intervals_with_timestamps = find_downfall_intervals_with_timestamps(oi_values, cumulative_threshold, change_threshold, tolerance)
    
    if price_index=="avg":
        price_values=[(float(kline[1])+float(kline[4]))/2 for kline in klines_data]
    else:
        price_values = [kline[price_index] for kline in klines_data] 
    price_downfall_intervals_with_timestamps = find_downfall_intervals_with_timestamps(price_values, cumulative_threshold, change_threshold, tolerance)
    
    # aggregate corresponding intervals into one more precise
    downfall_intervals=[] 
    if len(oi_downfall_intervals_with_timestamps)!=0 and len(price_downfall_intervals_with_timestamps)!=0:
        for oi_change, start_index1, end_index1 in oi_downfall_intervals_with_timestamps:
            # for each oi downfall interval, find downfall interval
            for price_change, start_index2, end_index2 in price_downfall_intervals_with_timestamps:
                if abs(start_index1-start_index2)<=tolerance*2 or abs(end_index1-end_index2)<=tolerance*2:
                    start_index=max(start_index1,start_index2)
                    end_index=min(end_index1,end_index2)
                    
                    # if not appropriate interval, skip
                    if start_index>=end_index:
                        continue
                    downfall_intervals.append((start_index,end_index))
                    break
    return downfall_intervals

def find_dump_intervals(symbol='BTCUSDT',period='5m',tolerance=4):
    # finds dumps intervals from current moment till month ago, returns list of tuples (start_unix,end_unix)
    now = datetime.now()
    time_delta = timedelta(minutes=now.minute % 5)
    rounded_time = (now - time_delta).replace(second=0, microsecond=0)
    rounded_unix_time = int(rounded_time.timestamp())*1000

    timestamps=[rounded_unix_time-150000000*i for i in range(17)] # Every period is 150000seconds=41.6hours=5min*500 (so no conflict with oi data receving)
    timestamps.sort()

    dumps_intervals=[]
    for i in range (1,len(timestamps)):
        start_timestamp=timestamps[i-1]
        end_timestamp=timestamps[i]
        downfall_intervals=find_dumps(symbol,period,price_index=2,change_threshold=-0.002,cumulative_threshold=0.015,tolerance=tolerance,start_time=start_timestamp,end_time=end_timestamp)
        if len(downfall_intervals)>0:
            for start_index, end_index in downfall_intervals:
                start_unix=convert_index_to_unix(start_timestamp,period,start_index)
                end_unix=convert_index_to_unix(start_timestamp,period,end_index)
                
                #include last period
                if period=='5m':
                    end_unix+=300_000
                    
                dumps_intervals.append((start_unix,end_unix))
    return dumps_intervals

if __name__ == "__main__":
    print(find_dump_intervals())