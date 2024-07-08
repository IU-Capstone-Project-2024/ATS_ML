def testOrder_reward(testOrder,df):
    # calculates reward for testOrder object
    reward=0
    
    # df contains rows with 15 second aggregated data
    rows_after_buy=df[df['unix']>=testOrder.entry_timestamp]
    
    max_price=max(rows_after_buy['Close'])
    min_price=min(rows_after_buy['Close'])
    max_potential_profit=calc_profit(testOrder.entry_price,max_price)-0.1
    min_potential_loss=calc_profit(testOrder.entry_price,min_price)-0.1
    
    # if bought on high, fix further division by 0 and exploding gradient
    if max_potential_profit<=0.1:
        max_potential_profit= -0.1
    
    # potential Profit/Loss ratio=[bad 0..1..7 good]
    if abs(max_potential_profit/min_potential_loss)<=1:
        reward-=1/abs(max_potential_profit/min_potential_loss) #[-20..-1]
    else:
        reward+=abs(max_potential_profit/min_potential_loss) # [1..10]
    
    real_profitloss=calc_profit(testOrder.entry_price,testOrder.exit_price)
    
    # action sell was performed
    if real_profitloss!=None:
        testOrder.profit=real_profitloss-0.1
        reward+=testOrder.profit*60 # [-60..30]
        time_in_trade=(testOrder.exit_timestamp-testOrder.entry_timestamp)/1000
        # Penalty for too little time in trade
        if time_in_trade<=30:
            time_in_trade=600
    
    # only bought, but didnt sell
    else: 
        time_in_trade=300 # neutral about time in trade
        
        # keep testOrder.profit= unrealized profit 
        reward+=testOrder.profit*10 # [-10..5]
    
    # estimates time in trade. Preferably be in trade less than 300 seconds
    reward += (1-time_in_trade/300)*5 # [-10,5]
    
    # find rewards out of bounds
    if reward>=20 or reward<=-20:
        print(reward)
    return reward

def calc_profit(entry_price, exit_price, fee=0.1):
    if exit_price!=None and entry_price!=None:
        profitloss=(exit_price/entry_price-1)*100
        if profitloss<=-1 or profitloss>=1:
            pass
    else:
        # if bought but not sold
        profitloss=None
    
    return profitloss