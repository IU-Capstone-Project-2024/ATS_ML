import requests
import pandas as pd
import time
import sys
import argparse
import os
import subprocess


def get_binance_klines(symbol, interval, start_time, end_time):
    base_url = 'https://api.binance.com'
    endpoint = '/api/v3/klines'
    url = base_url + endpoint

    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': 1000
    }

    while True:
        try:
            response = requests.get(url, params=params)
            data = response.json()
            if 'code' in data:
                raise Exception(data['msg'])
            break
        except Exception as e:
            print(f"Error: {e}. Retrying in 5 seconds...")
            time.sleep(5)

    return data


def fetch_data_for_past_week(symbol, interval, hours=168):
    end_time = int(time.time() * 1000)
    start_time = end_time - (hours * 60 * 60 * 1000)

    all_data = []
    current_start_time = start_time

    print("Fetching data...")

    while current_start_time < end_time:
        data = get_binance_klines(symbol, interval, current_start_time, end_time)
        all_data.extend(data)
        if not data:
            break
        current_start_time = data[-1][0] + 1
        progress = (current_start_time - start_time) / (end_time - start_time)
        sys.stdout.write(f"\rProgress: [{'#' * int(progress * 50):<50}] {progress * 100:.2f}%")
        sys.stdout.flush()

    columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time',
               'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume',
               'Taker Buy Quote Asset Volume', 'Ignore']
    df = pd.DataFrame(all_data, columns=columns)

    # Convert timestamp to readable date
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')

    return df


def get_symbols_list():
    base_url = 'https://api.binance.com'
    endpoint = '/api/v3/exchangeInfo'
    url = base_url + endpoint

    response = requests.get(url)
    data = response.json()

    symbols = [symbol['symbol'] for symbol in data['symbols']]

    return symbols


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Binance Data Scraper')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Symbol to fetch data for')
    parser.add_argument('--interval', type=str, default='1m', help='Interval of data')
    parser.add_argument('--hours', type=int, default=24, help='Number of hours to look back')

    args = parser.parse_args()

    symbol = args.symbol
    interval = args.interval
    hours = args.hours

    df = fetch_data_for_past_week(symbol, interval, hours)
    print("\nData fetching complete.")
    # Save to CSV for further analysis

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '../../..', 'data', f'{symbol}_{interval}', 'raw')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{symbol}_{interval}_data.csv')
    df.to_csv(output_file, index=False)

    # Notify DVC about changes
    subprocess.run(['dvc', 'add', output_file], check=True)
    subprocess.run(['dvc', 'commit', output_file], check=True)
