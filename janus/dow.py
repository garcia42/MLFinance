import pandas as pd
import numpy as np
from scipy.signal import find_peaks
pd.options.mode.chained_assignment = None  # default='warn'

def calculate_equity_curve_dow_theory(trades_df: pd.DataFrame, initial_capital=50000, 
                                      use_position_sizing=False, risk_percentage=2):
    """
    Calculate equity curve using trading method based on peaks and troughs identified with scipy
    
    Parameters:
    - trades_df: DataFrame with OHLC data
    - initial_capital: Starting capital
    - risk_percentage: Percentage of capital to risk per trade
    
    Returns:
    - DataFrame with equity curve and trade statistics
    """
    # Ensure dates are sorted
    trades_df = trades_df.sort_index()
    
    # Create a copy of price data for analysis
    df = trades_df.copy()
    
    # Initialize columns for signals and tracking
    df['signal'] = 0  # 1 for buy, -1 for sell/short
    df['equity'] = initial_capital if use_position_sizing else 0
    
    # Find all peaks and troughs using scipy's find_peaks
    # For peaks, we look for points where the value is higher than neighboring points
    peaks, _ = find_peaks(df['High'].values, distance=4)  # Adjust distance parameter as needed
    
    # For troughs, we invert the data and find peaks in the inverted data
    troughs, _ = find_peaks(-df['Low'].values, distance=4)  # Adjust distance parameter as needed
    
    # Add columns to mark peaks and troughs
    df['swing_high'] = np.nan
    df['swing_low'] = np.nan
    
    # Mark the swing highs and lows in the dataframe
    for peak in peaks:
        df.iloc[peak, df.columns.get_loc('swing_high')] = df['High'].iloc[peak]
        
    for trough in troughs:
        df.iloc[trough, df.columns.get_loc('swing_low')] = df['Low'].iloc[trough]
    
    # Initialize variables for tracking swings
    last_swing_high_price = None
    last_swing_low_price = None
    entry_price = None
    entry_date = None
    current_position = None
    last_signal_was_loss = True
    current_trend = None
    
    # Get the initial swing high and low from the first identified peaks and troughs
    if len(peaks) > 0:
        last_swing_high_price = df['High'].iloc[peaks[0]]
    else:
        last_swing_high_price = df['High'].iloc[0]
        
    if len(troughs) > 0:
        last_swing_low_price = df['Low'].iloc[troughs[0]]
    else:
        last_swing_low_price = df['Low'].iloc[0]
    
    # Track trade statistics
    trades = []
    
    # Process data in a single pass
    for i in range(len(df)):
        # Skip first few days for stability
        if i <= 2:
            continue
            
        current_idx = df.index[i]
        current_price = df['Close'].iloc[i]
        
        # Update last swing high and low as we encounter them
        if i in peaks:
            last_swing_high_price = df['High'].iloc[i]
                
        if i in troughs:
            last_swing_low_price = df['Low'].iloc[i]
        
        # Check if higher high or lower low happened today
        new_trend_long = df['High'].iloc[i] > last_swing_high_price and current_trend != 'long'
        new_trend_short = df['Low'].iloc[i] < last_swing_low_price and current_trend != 'short'
        
        if new_trend_long or new_trend_short:
            current_trend = 'long' if new_trend_long else 'short'
            exit_price = current_price
            previous_entry_price = entry_price

            # Exit position if in one because of a new trend direction
            if current_position is not None:
                df.iloc[entry_date:i, df.columns.get_loc('signal')] = 1 if current_position == 'long' else -1
                trade_profit = exit_price - entry_price if current_position == 'long' else entry_price - exit_price
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_idx,
                    'position': current_position,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit': trade_profit,
                    'profit_pct': (trade_profit / entry_price) * 100,
                })
            current_position = None
            
            # Enter new position if previous signal was a loss
            entry_price = current_price
            entry_date = i
            
            # Check if the previous trade was a loss, compare current exit price to previous entry
            if previous_entry_price is not None:
                trade_profit = exit_price - previous_entry_price if new_trend_short else previous_entry_price - exit_price
                # last_signal_was_loss = trade_profit < 0
                # if last_signal_was_loss: # Entering position
                current_position = 'long' if new_trend_long else 'short'
    
    # Close final position at the end of the data if still open
    if current_position is not None:
        df.iloc[entry_date:i, df.columns.get_loc('signal')] = -1 if current_position == 'short' else 1
        final_price = df['Close'].iloc[-1]
        exit_date = df.index[-1]
        trade_profit = (final_price - entry_price) if current_position == 'long' else (entry_price - final_price)
        trades.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'position': current_position,
            'entry_price': entry_price,
            'exit_price': final_price,
            'profit': trade_profit,
            'profit_pct': (trade_profit / entry_price) * 100
        })
    
    # Convert trades list to DataFrame for analysis
    trades_df = pd.DataFrame(trades)
    
    # Calculate trade statistics
    if len(trades_df) > 0:
        win_rate = (trades_df['profit'] > 0).mean() * 100
        avg_win = trades_df.loc[trades_df['profit'] > 0, 'profit'].mean() if sum(trades_df['profit'] > 0) > 0 else 0
        avg_loss = trades_df.loc[trades_df['profit'] < 0, 'profit'].mean() if sum(trades_df['profit'] < 0) > 0 else 0
        avg_profit = trades_df['profit'].mean()
        avg_profit_pct = trades_df['profit_pct'].mean()
        avg_loss_profit_pct = trades_df.loc[trades_df['profit'] < 0, 'profit_pct'].mean()
        avg_win_profit_pct = trades_df.loc[trades_df['profit'] > 0, 'profit_pct'].mean()
        total_profit = trades_df['profit'].sum()
        profit_factor = abs(trades_df.loc[trades_df['profit'] > 0, 'profit'].sum() / 
                          trades_df.loc[trades_df['profit'] < 0, 'profit'].sum()) if sum(trades_df['profit'] < 0) < 0 else float('inf')
        
        print("\nTrade Statistics:")
        print(f"Total trades: {len(trades_df)}")
        print(f"Win rate: {win_rate:.1f}%")
        print(f"Average win: ${avg_win:.2f}")
        print(f"Average loss: ${avg_loss:.2f}")
        print(f"Average profit per trade: ${avg_profit:.2f}")
        print(f"Average profit percent per trade: {avg_profit_pct:.2f}%")
        print(f"Average loss profit percent per trade: {avg_loss_profit_pct:.2f}%")
        print(f"Average win profit percent per trade: {avg_win_profit_pct:.2f}%")
        print(f"Total profit: ${total_profit:.2f}")
        print(f"Profit factor: {profit_factor:.2f}")
    
    # Calculate daily returns and equity curves
    df['trade_profit'] = 0
    
    # Fixed contract approach (1 contract)
    df['contract_return'] = df['Close'].diff() * df['signal'].shift(1)
    df.loc[df.index[0], 'contract_return'] = 0
    df['contract_equity'] = df['contract_return'].cumsum()
    
    # Use fixed contract equity (starting from 0)
    df['equity'] = df['contract_equity']

    return df, trades_df

async def close_non_leader_positions(leaders, actions):
    """Close all positions that are not in the leaders list"""
    # Get current portfolio positions
    portfolio = await ib_client.get_positions()
    
    if not portfolio:
        logger.info("No positions found to process")
        return
        
    for position in portfolio:
        symbol = position.contract.symbol
        if symbol not in leaders:
            try:
                await close_position(position, symbol, actions)
            except Exception as e:
                logger.exception(f"Failed to close position for {symbol}")


async def close_position(position, symbol, actions):
    """Close a single position and record the action"""
    # Validate contract first
    contract = await ib_client.validate_contract(symbol)
    quantity = abs(position.position)
    
    # Create opposite order (sell if long, buy if short)
    side = 'SELL' if position.position > 0 else 'BUY'
    order = MarketOrder(side, quantity)
    
    # Submit the order
    trade = await ib_client.place_order(contract, order)

    # Wait for order with timeout
    try:
        await wait_for_order(trade)
    except TimeoutError as e:
        logger.error(f"Order timeout for {symbol}: {e}")
        return
    except Exception as e:
        logger.exception(f"Unexpected error waiting for order for {symbol}")
        return

    # Check for complete fill
    if trade.orderStatus.filled != quantity:
        logger.warning(f"Partial fill for {symbol}: {trade.orderStatus.filled}/{quantity}")
    
    # Record action
    actions["positions_closed"].append({
        "symbol": symbol,
        "quantity": quantity,
        "filled_quantity": trade.orderStatus.filled,
        "side": side,
        "status": trade.orderStatus.status,
        "avg_fill_price": trade.orderStatus.avgFillPrice
    })
    
    logger.info(f"Closed position for {symbol}: {trade.orderStatus.filled} shares {side} at {trade.orderStatus.avgFillPrice}")


async def execute_leader_buys(leaders, actions):
    """Execute buy orders for all leader symbols"""
    # Get available funds
    available_cash = await get_available_funds()
    logger.info(f"Available cash: ${available_cash:,.2f}")
    
    # Calculate allocation per position
    cash_per_position = calculate_allocation_per_position(available_cash, leaders)
    if cash_per_position <= 0:
        raise HTTPException(status_code=400, detail="Insufficient funds after accounting for commissions")
    
    logger.info(f"Allocating ${cash_per_position:,.2f} per position")
    
    # Place buy orders for each leader
    for symbol in leaders:
        try:
            await buy_leader_position(symbol, cash_per_position, actions)
        except Exception as e:
            logger.exception(f"Error buying {symbol}")
            actions["buys_executed"].append({
                "symbol": symbol,
                "error": str(e),
                "traceback": traceback.format_exc() if DEBUG else None
            })


async def get_available_funds():
    """Get the current available funds from account"""
    account_summary = await ib_client.get_account_summary()
    return float(account_summary.get("AvailableFunds", 0))


def calculate_allocation_per_position(available_cash, leaders):
    """Calculate how much cash to allocate per position"""
    ESTIMATED_COMMISSION_PER_TRADE = 1.00  # Adjust based on your broker
    total_commission = len(leaders) * ESTIMATED_COMMISSION_PER_TRADE
    adjusted_cash = available_cash - total_commission
    
    # Only spend up to 30k
    adjusted_cash = min(30000, adjusted_cash)
    
    if adjusted_cash <= 0:
        return 0
        
    return adjusted_cash / len(leaders)


async def buy_leader_position(symbol, cash_per_position, actions):
    """Buy a single leader position and record the action"""
    # Validate contract
    contract = await ib_client.validate_contract(symbol)
    
    # Calculate shares to buy
    shares = await ib_client.calculate_shares_to_buy(symbol, cash_per_position)
    
    if shares <= 0:
        logger.warning(f"Insufficient funds to buy {symbol} at current price")
        return
    
    logger.info("Buying %d %s", shares, symbol)
    
    # Create and place buy order
    order = MarketOrder('BUY', shares)
    trade = await ib_client.place_order(contract, order)
    
    logger.info("Waiting for order for %s", symbol)
    
    # Wait for order with timeout
    try:
        await wait_for_order(trade)
        # Check result after wait_for_order
        if trade.orderStatus.status == "Filled":
            record_successful_buy(symbol, shares, trade, actions)
        else:
            logger.error(f"Order not fully filled for {symbol}: {trade.orderStatus.status}")
            
    except TimeoutError as e:
        logger.error(f"Order timeout for {symbol}: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error waiting for order for {symbol}")


def record_successful_buy(symbol, shares, trade, actions):
    """Record a successful buy execution"""
    actions["buys_executed"].append({
        "symbol": symbol,
        "shares": shares,
        "filled_shares": trade.orderStatus.filled,
        "status": trade.orderStatus.status,
        "avg_fill_price": trade.orderStatus.avgFillPrice
    })
    
    logger.info(f"Bought {trade.orderStatus.filled} shares of {symbol} at {trade.orderStatus.avgFillPrice}")


def handle_exception(message, exception):
    """Handle and log exceptions"""
    error_trace = traceback.format_exc()
    logger.error(f"{message}: {exception}")
    logger.error(f"Traceback: {error_trace}")