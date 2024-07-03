def testOrder_reward(testOrder):
    # calculates reward for testOrder object
    
    reward=0
    real_profitloss=calc_profit(testOrder.entry_price,testOrder.exit_price)
    
    # action sell was performed
    if real_profitloss!=None:
        testOrder.profit=real_profitloss
        reward+=testOrder.profit*10
        time_in_trade=(testOrder.exit_timestamp-testOrder.entry_timestamp)/1000
    
    # only bought, but didnt sell
    else: 
        time_in_trade=300 # neutral about time in trade
        
        # keep testOrder.profit= unrealized profit 
        reward+=testOrder.profit*5
    
    # estimates time in trade. Preferably be in trade less than 300 seconds
    reward= reward + (0.5-time_in_trade/600)
    
    # find rewards out of bounds
    if reward<=-1 or reward>=2:
        pass
    return reward

def calc_profit(entry_price, exit_price):
    if exit_price!=None and entry_price!=None:
        profitloss=(exit_price/entry_price-1)*100
        if profitloss<=-1 or profitloss>=1:
            pass
    else:
        # if bought but not sold
        profitloss=None
        
    return profitloss