class TestOrder:
    def __init__(self, symbol, entry_price, entry_timestamp, period, strategy):
        self.symbol=symbol
        self.entry_price=entry_price
        self.entry_timestamp=entry_timestamp
        self.period=period
        self.strategy=strategy
        
        self.exit_timestamp=None
        self.exit_price=None
        self.profit=0
    
# TODO: bybit client real orders
