
import pandas as pd
from ib_insync import *
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def get_futures_data(ib: IB, symbol='GC', exchange='COMEX'):
    """Fetch daily gold futures data for analysis using continuous contract"""
    try:
        # Create a continuous futures contract
        # Use '@' symbol to indicate a continuous contract
        gold_contract = ContFuture(symbol=symbol, exchange=exchange)
        
        # No need to qualify a continuous contract
        
        # Request daily historical data (1 year)
        bars = await ib.reqHistoricalDataAsync(
            gold_contract,
            endDateTime='',  # Current time
            durationStr='365 D',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True
        )
        
        if not bars:
            logger.warning("No historical data received for gold futures")
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = util.df(bars)
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        return df
        
    except Exception as e:
        logger.exception(f"Error retrieving gold futures data: {e}")
        return pd.DataFrame()

def place_continuous_futures_trade(ib: IB, symbol='GC', exchange='COMEX', 
                                        action='BUY', quantity=1) -> Trade:
    """Place a trade using the continuous futures contract"""
    try:
        # Create continuous futures contract
        cont_contract = ContFuture(symbol=symbol, exchange=exchange)
        
        # Create market order
        order = MarketOrder(action=action, totalQuantity=quantity)
        
        # Place the order
        trade = ib.placeOrder(cont_contract, order)
        logger.info(f"Placed {action} order for {quantity} {symbol} continuous contract")
        
        return trade
        
    except Exception as e:
        logger.exception(f"Error placing continuous futures trade: {e}")
        return None