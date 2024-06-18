Here we store some scripts that can be used to access data for training or inference.

Right now we have only binance_data_scraper.py

The usage is simple:
- Run the python script with arguments TOKEN_SYMBOL, interval, last_hours
- It will get the data for the specified token in last hours with given step inteval
- After that data will be stored in ../../../data/TOKEN_SYMBOL_interval/raw in a csv format
- After saving the data DVC will be notified about changes