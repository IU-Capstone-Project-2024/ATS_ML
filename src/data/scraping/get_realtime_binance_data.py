import asyncio
from ATS_ML.src.data.scraping.get_historical_binance_data import listen_to_futures_stream
from ATS_ML.src.data.scraping.get_historical_binance_data import listen_to_spot_stream

# Data by second via websocket:
# delta spot
# Oi change% futures through restapi
# Amount trades futures
# Liqudation change futures%
# Volume change spot%
# Stakan plotnost futures+spot, Round numbers

async def main(symbol, stream_string):
    received_data = []  # Initialize an empty list to save received data
    async for data in listen_to_futures_stream(symbol, stream_string):
        print(data)
        received_data.append(data)  # Save each received data to the list
    print(f"All received data: {received_data}")  # Print all received data

# Run the main function
symbol = "btcusdt"
stream_string = "markPrice"
asyncio.get_event_loop().run_until_complete(main(symbol, stream_string))
