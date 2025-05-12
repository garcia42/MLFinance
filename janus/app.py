from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from ib_insync import IB, Stock, Contract, MarketOrder, Trade
import os
import signal
import sys
import uvicorn
from typing import Dict, List, Optional, Annotated
import asyncio
import logging
from pydantic import BaseModel
import traceback
from datetime import datetime
from pytz import timezone
import time
from math import isnan
from contextlib import asynccontextmanager
# from trade_visualization import create_trade_visualization

# Import your updated IBClient
from ib_client import IBClient
from leaders import get_positive_leaders

import dow
import future_data

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set debug mode from environment
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

# Global IB client
ib_client = IBClient()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connect to IB Gateway at startup
    print("Starting up - connecting to IB Gateway...")
    try:
        await ib_client.ensure_connected()
    except Exception as e:
        print(f"Failed to establish initial connection: {e}")
    
    # Yield control back to FastAPI
    yield
    
    # Disconnect on shutdown
    print("Shutting down - disconnecting from IB Gateway...")
    if ib_client.ib.isConnected():
        # The disconnect method is synchronous, not async, so don't use await
        ib_client.ib.disconnect()
    ib_client.executor.shutdown(wait=False)
    print("Disconnected from IB Gateway")

# Use the lifespan manager with your FastAPI app
app = FastAPI(lifespan=lifespan)

class Position(BaseModel):
    contract: Dict
    position: float
    avgCost: float

class AccountValue(BaseModel):
    tag: str
    value: str
    currency: str
    account: str

class MarketData(BaseModel):
    symbol: str
    last: float
    bid: float
    ask: float
    volume: int
    close: float

# Dependency to get a connected IB client
async def get_ib() -> IB:
    return await ib_client.ensure_connected()

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    error_detail = str(exc)
    if DEBUG:
        error_detail = f"{str(exc)}\n\nTraceback:\n{traceback.format_exc()}"
    logger.error(f"Unhandled exception: {error_detail}")
    return JSONResponse(
        status_code=500,
        content={"detail": error_detail}
    )

@app.get("/status")
async def get_status(ib: Annotated[IB, Depends(get_ib)]):
    """Get the connection status with detailed diagnostics"""
    status = {
        'connected': ib.isConnected(),
        'client_id': ib_client.ib_config['clientId'],
        'mode': 'paper' if ib_client.ib_config['port'] == 4002 else 'live',
        'diagnostics': {}
    }
    
    if status['connected']:
        try:
            contract = Stock('SPY', 'SMART', 'USD')
            status['diagnostics']['contract'] = {
                'symbol': 'SPY',
                'exchange': 'SMART',
                'currency': 'USD'
            }
            
            try:
                qualified = await ib.qualifyContractsAsync(contract)
                status['diagnostics']['contract_qualification'] = {
                    'working': True,
                    'qualified': bool(qualified)
                }
            except asyncio.TimeoutError:
                status['diagnostics']['contract_qualification'] = {
                    'working': False,
                    'error': 'Timeout qualifying contract'
                }
                return status
            except Exception as e:
                status['diagnostics']['contract_qualification'] = {
                    'working': False,
                    'error': str(e),
                    'traceback': traceback.format_exc() if DEBUG else None
                }
                return status
            
            if qualified:
                try:
                    bars = await ib.reqHistoricalDataAsync(
                        contract,
                        endDateTime='',
                        durationStr='1 D',
                        barSizeSetting='1 min',
                        whatToShow='TRADES',
                        useRTH=True
                    )
                    
                    status['historical_data'] = {
                        'working': len(bars) > 0,
                        'bars_received': len(bars),
                        'latest_timestamp': bars[-1].date.isoformat() if bars else None,
                        'first_timestamp': bars[0].date.isoformat() if bars else None
                    }
                except Exception as e:
                    status['historical_data'] = {
                        'working': False,
                        'error': str(e),
                        'traceback': traceback.format_exc() if DEBUG else None
                    }
        except Exception as e:
            status['error'] = str(e)
            status['traceback'] = traceback.format_exc() if DEBUG else None
    
    return status

@app.get("/positions", response_model=List[Position])
async def get_positions(ib: Annotated[IB, Depends(get_ib)]):
    """Get current positions"""
    try:
        # Use the IBClient method that handles async properly
        positions = await ib_client.get_positions()
        return [{
            'contract': {
                'symbol': pos.contract.symbol,
                'secType': pos.contract.secType,
                'exchange': pos.contract.exchange,
                'currency': pos.contract.currency
            },
            'position': pos.position,
            'avgCost': pos.avgCost
        } for pos in positions]
    except Exception as e:
        logger.exception("Error getting positions")
        raise HTTPException(
            status_code=500, 
            detail=str(e) if not DEBUG else f"{str(e)}\n\n{traceback.format_exc()}"
        )

@app.get("/account")
async def how_much(ib: Annotated[IB, Depends(get_ib)]):
    try:
        # Use IBClient's method to get account summary in a thread-safe way
        account_summary = await ib_client.get_account_summary()
        return account_summary
    except Exception as e:
        logger.exception("Error in account endpoint")
        raise HTTPException(
            status_code=500, 
            detail=str(e) if not DEBUG else f"{str(e)}\n\n{traceback.format_exc()}"
        )

async def wait_for_order(trade: Trade, timeout=30):
    """Wait for an order to complete with timeout"""
    try:
        start_time = time.time()
        while not trade.isDone():
            if time.time() - start_time > timeout:
                error_msg = f"Order did not complete within {timeout} seconds"
                logger.error(error_msg)
                raise TimeoutError(error_msg)
            await asyncio.sleep(1)
    except Exception as e:
        if not isinstance(e, TimeoutError):
            logger.exception(f"Error waiting for order: {trade.contract.symbol if trade.contract else 'Unknown'}")
        raise

def is_market_hours(current_time: datetime) -> bool:
    """Check if we're in regular market hours (9:30 AM - 4:00 PM ET, Mon-Fri)"""
    if current_time.weekday() > 4:  # Saturday = 5, Sunday = 6
        return False
    market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= current_time <= market_close

@app.get("/daily_futures")
async def do_futures_buys_for_day(ib: Annotated[IB, Depends(get_ib)]):
    """Execute daily orders - sell non-leaders and prepare to buy new leaders"""
    gold_symbol = "GC"
    try:
        # Get gold futures data and calculate signals
        gold_data = await ib_client.get_futures_data(symbol=gold_symbol, exchange="COMEX")
        if gold_data.empty:
            logger.warning("No gold futures data retrieved")
            return {"positions_closed": [], "signals": [], "trades_executed": []}
        print(gold_data)
        
        # # Create visualization
        # create_trade_visualization(gold_data, "Gold_data.csv")
        
        # # Get current leaders
        position_df, _ = dow.calculate_equity_curve_dow_theory(trades_df=gold_data, initial_capital=50000, use_position_sizing=False)
        
        print(position_df['signal'])
        
        current_signal = position_df['signal'].iloc[-1]
        
        logger.log(logging.INFO, f"Current signal: {current_signal}")
        
        front_contract = await ib_client.get_front_futures_contract(symbol=gold_symbol, exchange="COMEX")

        # Get current portfolio positions using the IBClient method
        portfolio = await ib_client.get_positions()

        # Identify any gold futures positions (could be in different contract months)
        gold_positions = [pos for pos in portfolio if pos.contract.symbol == 'GC' and pos.contract.secType == 'FUT']

        total_gold_position = sum(pos.position for pos in gold_positions)
        print(f"Current signal: {current_signal}, Total gold position: {total_gold_position}")

        if gold_positions:
            # For existing positions that aren't in the front contract, close them first
            for pos in gold_positions:
                if pos.contract.lastTradeDateOrContractMonth != front_contract.lastTradeDateOrContractMonth:
                    # Close the position in this contract
                    action = "SELL" if pos.position > 0 else "BUY"
                    quantity = abs(pos.position)
                    trade = ib_client.place_futures_trade(contract=pos.contract, symbol=gold_symbol, action=action, quantity=quantity)
                    print(f"Closing position in contract {pos.contract.lastTradeDateOrContractMonth}")

        # Now handle the new position in the front contract
        if total_gold_position > 0 and current_signal == -1:  # Net Long Position, Switch to Short
            # Exit long and enter short
            trade = ib_client.place_futures_trade(contract=front_contract, symbol=gold_symbol, action="SELL", quantity=abs(total_gold_position) + 1)
            
        elif total_gold_position < 0 and current_signal == 1:  # Net Short Position, Switch to Long
            # Exit short and enter long
            trade = ib_client.place_futures_trade(contract=front_contract, symbol=gold_symbol, action="BUY", quantity=abs(total_gold_position) + 1)
            
        elif total_gold_position == 0:  # No existing position
            if current_signal == 1:  # Go long
                trade = ib_client.place_futures_trade(contract=front_contract, symbol=gold_symbol, action="BUY", quantity=1)
            elif current_signal == -1:  # Go short
                trade = ib_client.place_futures_trade(contract=front_contract, symbol=gold_symbol, action="SELL", quantity=1)
        
        # Get current portfolio positions using the IBClient method
        portfolio = await ib_client.get_positions()
        print(f"Positions: {portfolio}")
        
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Error in daily execution: {e}")
        logger.error(f"Traceback: {error_trace}")
        raise HTTPException(
            status_code=500, 
            detail=str(e) if not DEBUG else f"{str(e)}\n\n{traceback.format_exc()}"
        )

@app.get("/daily")
async def do_stock_buys_for_day(ib: Annotated[IB, Depends(get_ib)]):
    """Execute daily orders - sell non-leaders and prepare to buy new leaders"""
    try:
        # Get current leaders
        leaders = await get_positive_leaders(ib)
        if not leaders:
            logger.warning("No leaders found for today")
            return {"positions_closed": [], "leaders_to_buy": [], "buys_executed": []}

        # Get current portfolio positions using the IBClient method
        portfolio = await ib_client.get_positions()
        
        # Track actions taken
        actions = {
            "positions_closed": [],
            "leaders_to_buy": leaders,
            "buys_executed": []
        }

        # Get initial account summary
        account_summary = await ib_client.get_account_summary()
        # Parse account summary to get available funds
        available_funds = float(account_summary.get("AvailableFunds", 0))
        
        if available_funds <= 0:
            raise HTTPException(status_code=400, detail="Insufficient funds in account")

        # Process each position
        if portfolio:
            for position in portfolio:
                symbol = position.contract.symbol
                if symbol not in leaders:
                    try:
                        # Validate contract first using IBClient method
                        contract = await ib_client.validate_contract(symbol)
                        quantity = abs(position.position)
                        
                        # Create opposite order (sell if long, buy if short)
                        side = 'SELL' if position.position > 0 else 'BUY'
                        order = MarketOrder(side, quantity)
                        
                        # Submit the order using IBClient method
                        trade = await ib_client.place_order(contract, order)

                        # Wait for order with timeout
                        try:
                            await wait_for_order(trade)
                        except TimeoutError as e:
                            logger.error(f"Order timeout for {symbol}: {e}")
                            continue
                        except Exception as e:
                            logger.exception(f"Unexpected error waiting for order for {symbol}")
                            continue

                        # Check for complete fill
                        if trade.orderStatus.filled != quantity:
                            logger.warning(f"Partial fill for {symbol}: {trade.orderStatus.filled}/{quantity}")
                        
                        actions["positions_closed"].append({
                            "symbol": symbol,
                            "quantity": quantity,
                            "filled_quantity": trade.orderStatus.filled,
                            "side": side,
                            "status": trade.orderStatus.status,
                            "avg_fill_price": trade.orderStatus.avgFillPrice
                        })
                        
                        logger.info(f"Closed position for {symbol}: {trade.orderStatus.filled} shares {side} at {trade.orderStatus.avgFillPrice}")
                    
                    except Exception as e:
                        logger.exception(f"Failed to close position for {symbol}")
                        continue
        else:
            logger.info("No positions found to process")

        # Get available cash after closing positions
        updated_account_summary = await ib_client.get_account_summary()
        # Parse updated account summary to get available funds
        available_cash = float(updated_account_summary.get("AvailableFunds", 0))
        
        logger.info(f"Available cash: ${available_cash:,.2f}")
        
        # Calculate cash per position accounting for commissions
        ESTIMATED_COMMISSION_PER_TRADE = 1.00  # Adjust based on your broker
        total_commission = len(leaders) * ESTIMATED_COMMISSION_PER_TRADE
        adjusted_cash = available_cash - total_commission
        
        # Only spend up to 30k
        adjusted_cash = min(30000, adjusted_cash)

        if adjusted_cash <= 0:
            raise HTTPException(status_code=400, detail="Insufficient funds after accounting for commissions")
            
        cash_per_position = adjusted_cash / len(leaders)
        logger.info(f"Allocating ${cash_per_position:,.2f} per position")
        
        # Place buy orders for each leader
        for symbol in leaders:
            try:
                # Validate contract using IBClient method
                contract = await ib_client.validate_contract(symbol)
                
                # Calculate shares to buy
                shares = await ib_client.calculate_shares_to_buy(symbol, cash_per_position)
                
                if shares <= 0:
                    logger.warning(f"Insufficient funds to buy {symbol} at current price")
                    continue
                
                logger.info("Buying %d %s", shares, symbol)
                
                # Create and place buy order using IBClient method
                order = MarketOrder('BUY', shares)
                trade = await ib_client.place_order(contract, order)
                
                logger.info("Waiting for order for %s", symbol)
                
                # Wait for order with timeout
                try:
                    await wait_for_order(trade)
                    # Checking the result after wait_for_order
                    if trade.orderStatus.status == "Filled":
                        logger.info("Order was filled for %s and units %d", symbol, shares)
                        
                        actions["buys_executed"].append({
                            "symbol": symbol,
                            "shares": shares,
                            "filled_shares": trade.orderStatus.filled,
                            "status": trade.orderStatus.status,
                            "avg_fill_price": trade.orderStatus.avgFillPrice
                        })
                        
                        logger.info(f"Bought {trade.orderStatus.filled} shares of {symbol} at {trade.orderStatus.avgFillPrice}")
                    else:
                        logger.error(f"Order not fully filled for {symbol}: {trade.orderStatus.status}")
                        
                except TimeoutError as e:
                    logger.error(f"Order timeout for {symbol}: {e}")
                    continue
                except Exception as e:
                    logger.exception(f"Unexpected error waiting for order for {symbol}")
                    continue
                
            except Exception as e:
                logger.exception(f"Error buying {symbol}")
                actions["buys_executed"].append({
                    "symbol": symbol,
                    "error": str(e),
                    "traceback": traceback.format_exc() if DEBUG else None
                })
        
        return actions
        
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Error in daily execution: {e}")
        logger.error(f"Traceback: {error_trace}")
        raise HTTPException(
            status_code=500, 
            detail=str(e) if not DEBUG else f"{str(e)}\n\n{error_trace}"
        )

# Set up proper signal handlers
def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown"""
    def shutdown_handler(signum, frame):
        print(f"Received shutdown signal {signum}")
        # Only set the stop flag, don't force exit
        if uvicorn_server:
            uvicorn_server.should_exit = True
        
    # Register signal handlers
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

if __name__ == '__main__':
    try:
        # Setup signal handlers
        setup_signal_handlers()

        port = int(os.getenv('PORT', '8080'))
        # Store the server instance to control shutdown
        uvicorn_server = uvicorn.Server(
            config=uvicorn.Config(
                app=app,
                host="0.0.0.0",
                port=port,
                log_level="info"
            )
        )
        # Use run instead of the simpler uvicorn.run
        uvicorn_server.run()
    except Exception as e:
        print(f'Fatal error: {e}')
        print(f'Traceback: {traceback.format_exc()}')