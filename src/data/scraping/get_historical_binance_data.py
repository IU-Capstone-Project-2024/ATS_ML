import websockets
import json
import asyncio
import requests
from datetime import datetime, timedelta

async def listen_to_futures_stream(symbol, stream_string):
    # Current data
    
    url = f"wss://fstream.binance.com/ws/{symbol}@{stream_string}"

    async with websockets.connect(url) as websocket:
        print("Connected to Binance WebSocket")

        async def send_pong():
            while True:
                await asyncio.sleep(180)  # Wait for 3 minutes
                try:
                    pong_message = json.dumps({"pong": "pong"})
                    await websocket.send(pong_message)
                    print("Sent pong message")
                except Exception as e:
                    print(f"Failed to send pong message: {e}")
                    break
                
        pong_task = asyncio.create_task(send_pong())
        
        # Decide when to unsubscribe
        async def unsubscribe():
            unsubscribe_message = {
                "method": "UNSUBSCRIBE",
                "params": [
                    f"{symbol}@{stream_string}"
                ],
                "id": 1
            }
            await websocket.send(json.dumps(unsubscribe_message))
            print("Unsubscribed from stream")

        try:
            while True:
                message = await websocket.recv()
                try:
                    data = json.loads(message)
                    yield data['p']
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")

        except websockets.ConnectionClosed:
            print("Connection closed")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            pong_task.cancel()

async def listen_to_spot_stream(symbol, stream_string):
    url = f"wss://stream.binance.com:9443/ws/{symbol}@{stream_string}"

    async with websockets.connect(url) as websocket:
        print("Connected to Binance WebSocket")

        buy_volume = 0
        sell_volume = 0
        last_reset_time = datetime.utcnow()

        async def send_pong():
            while True:
                await asyncio.sleep(180)  # Wait for 3 minutes
                try:
                    pong_message = json.dumps({"pong": "pong"})
                    await websocket.send(pong_message)
                    print("Sent pong message")
                except Exception as e:
                    print(f"Failed to send pong message: {e}")
                    break

        pong_task = asyncio.create_task(send_pong())

        async def unsubscribe():
            unsubscribe_message = {
                "method": "UNSUBSCRIBE",
                "params": [
                    f"{symbol}@{stream_string}"
                ],
                "id": 1
            }
            await websocket.send(json.dumps(unsubscribe_message))
            print("Unsubscribed from stream")

        try:
            while True:
                message = await websocket.recv()
                trade = json.loads(message)

                trade_volume = float(trade['q'])  # Quantity
                is_buyer_maker = trade['m']       # True if buyer is maker (implies sell market order)

                if is_buyer_maker:
                    sell_volume += trade_volume
                else:
                    buy_volume += trade_volume

                current_time = datetime.utcnow()
                if current_time - last_reset_time >= timedelta(seconds=1):
                    delta_volume = buy_volume - sell_volume
                    yield {
                        "time": current_time.isoformat(),
                        "delta_volume": delta_volume,
                        "buy_volume": buy_volume,
                        "sell_volume": sell_volume
                    }
                    # Reset volumes and timer
                    buy_volume = 0
                    sell_volume = 0
                    last_reset_time = current_time

        except websockets.ConnectionClosed:
            print("Connection closed")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            pong_task.cancel()  # Cancel the pong task when done

def get_klines_futures(symbol, period, limit=None, start_time=None, end_time=None):
    # if first time asked, get from api
    base_url = "https://fapi.binance.com"
    endpoint = "/fapi/v1/klines"
    params = {
        'symbol': symbol,
        'interval': period,
    }
    if limit:
        params['limit']=limit
    if start_time:
        params['startTime']=start_time
    if end_time:
        params['endTime']=end_time
    
    # average response time ~1s
    response = requests.get(base_url + endpoint, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()

def get_few_klines_spot(symbol, period, start_time=None, end_time=None, limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': period,
        'limit': limit
    }
    if start_time:
        params['startTime'] = start_time
    if end_time:
        params['endTime'] = end_time
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()
        
def get_full_klines_spot(symbol, period, start_time, end_time):
    full_klines=[]
    unix_traveler=start_time
    while unix_traveler<end_time:
        few_klines=get_few_klines_spot(symbol, period, unix_traveler,end_time)
        full_klines+=few_klines
        unix_traveler=full_klines[-1][6]
    return full_klines

def get_oi(symbol, period, limit=500, start_time=None,end_time=None):    
    # if first time asked, get from api
    base_url = "https://fapi.binance.com"
    endpoint = "/futures/data/openInterestHist"
    params = {
        'symbol': symbol,
        'period': period,
    }
    if limit:
        params['limit']=limit
    if start_time:
        params['startTime']=start_time
    if end_time:
        params['endTime']=end_time
        
    response = requests.get(base_url + endpoint, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()
        
def get_few_aggregated_spot_trades(symbol, start_time, end_time):
    # little portion from this timestamps
    base_url = "https://api.binance.com"
    endpoint = "/api/v3/aggTrades"
    params = {
        "symbol": symbol,
        "startTime": start_time,
        "endTime": end_time,
        "limit": 1000
    }

    response = requests.get(base_url + endpoint, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching data: {response.status_code} - {response.text}")
    
def get_full_aggtrades_spot(symbol,start_time,end_time):
    full_trades=[]
    unix_traveler=start_time
    while unix_traveler<end_time:
        few_trades=get_few_aggregated_spot_trades(symbol,unix_traveler,end_time)
        full_trades+=few_trades
        unix_traveler=full_trades[-1]['T']
        print(unix_traveler, len(few_trades))
    return full_trades