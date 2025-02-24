from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from ib_insync import IB, Stock, Contract, MarketOrder, Trade
import threading
import os
import signal
import sys
import uvicorn
from typing import Dict, List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
import nest_asyncio
import logging
from leaders import get_positive_leaders
import traceback

from datetime import datetime
from pytz import timezone
import time
import asyncio
from typing import Dict, List
from fastapi import HTTPException
from ib_insync import Stock, MarketOrder
from math import isnan

nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set debug mode from environment
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

class IBClient:
    def __init__(self):
        self.ib = IB()
        self.ib_config = {
            'host': os.getenv('IB_HOST', '127.0.0.1'),
            'port': int(os.getenv('IB_PORT', '4002')),  # 4001 for live, 4002 for paper
            'clientId': int(os.getenv('IB_CLIENT_ID', '1')),
        }
        self.connection_thread = None
        self.running = False
        self.loop = None
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Set up signal handlers
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f'Received signal {signum}. Starting shutdown...')
        self.running = False
        try:
            if self.ib.isConnected():
                asyncio.run_coroutine_threadsafe(self.ib.disconnect(), self.loop)
                self.executor.shutdown(wait=True)
        except Exception as e:
            print(f'Error during shutdown: {e}')
            print(f'Traceback: {traceback.format_exc()}')
        finally:
            sys.exit(0)

    async def connect_ib(self):
        """Async function to connect to IB Gateway"""
        try:
            await self.ib.connectAsync(**self.ib_config)
            print('Successfully connected to IB Gateway')
        except Exception as e:
            print(f'Connection error: {e}')
            print(f'Traceback: {traceback.format_exc()}')
            if 'port' in str(e).lower() or 'connection refused' in str(e).lower():
                print('Critical connection error, exiting...')
                sys.exit(1)

    async def maintain_connection(self):
        """Async function to maintain IB connection"""
        while self.running:
            if not self.ib.isConnected():
                print('Connecting to IB Gateway...')
                await self.connect_ib()
            await asyncio.sleep(1)

    def run_async_loop(self):
        """Run the async event loop in a separate thread and restart if needed"""
        while self.running:
            try:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                self.loop.run_until_complete(self.maintain_connection())
            except Exception as e:
                print(f'Event loop crashed: {e}')
                print(f'Traceback: {traceback.format_exc()}')
            finally:
                print("Event loop stopped. Restarting in 2 seconds...")
                time.sleep(2)  # Changed asyncio.sleep to time.sleep since this is not in an async context

    def start(self):
        """Start the connection maintenance thread"""
        if self.loop and self.loop.is_running():
            print("Event loop already running")
            return
        
        self.running = True
        self.connection_thread = threading.Thread(target=self.run_async_loop)
        self.connection_thread.daemon = True
        self.connection_thread.start()

    def stop(self):
        """Stop the connection maintenance thread"""
        self.running = False
        if self.loop:
            self.loop.stop()
        if self.connection_thread:
            self.connection_thread.join()
        if self.ib.isConnected():
            asyncio.run(self.ib.disconnect())
        self.executor.shutdown(wait=True)

from contextlib import asynccontextmanager

# Global IB client
ib_client = IBClient()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connect to IB Gateway at startup
    print("Starting up - connecting to IB Gateway...")
    try:
        await ib_client.ib.connectAsync(
            host=ib_client.ib_config['host'],
            port=ib_client.ib_config['port'],
            clientId=ib_client.ib_config['clientId']
        )
        print("Successfully connected to IB Gateway")
    except Exception as e:
        print(f"Failed to connect to IB Gateway: {e}")
        if 'port' in str(e).lower() or 'connection refused' in str(e).lower():
            print('Critical connection error')
            sys.exit(1)
    
    # Yield control back to FastAPI
    yield
    
    # Disconnect on shutdown
    print("Shutting down - disconnecting from IB Gateway...")
    if ib_client.ib.isConnected():
        await ib_client.ib.disconnect()
    ib_client.executor.shutdown(wait=False)
    print("Disconnected from IB Gateway")

# Now use the lifespan manager with your FastAPI app
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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/status")
async def get_status():
    """Get the connection status with detailed diagnostics"""
    status = {
        'connected': ib_client.ib.isConnected(),
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
                qualified = await ib_client.ib.qualifyContractsAsync(contract)
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
                    bars = await ib_client.ib.reqHistoricalDataAsync(
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
async def get_positions():
    """Get current positions"""
    if not ib_client.ib.isConnected():
        raise HTTPException(status_code=503, detail="Not connected to IB Gateway")
    
    try:
        positions = await ib_client.ib.reqPositionsAsync()
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

@app.get("/market_data/{symbol}", response_model=MarketData)
async def get_market_data(symbol: str):
    """Get market data for a symbol"""
    if not ib_client.ib.isConnected():
        raise HTTPException(status_code=503, detail="Not connected to IB Gateway")
    
    contract = Stock(symbol, 'SMART', 'USD')
    try:
        await ib_client.ib.qualifyContractsAsync(contract)
        [ticker] = await ib_client.ib.reqTickersAsync(contract)
        
        return {
            'symbol': symbol,
            'last': ticker.last,
            'bid': ticker.bid,
            'ask': ticker.ask,
            'volume': ticker.volume,
            'close': ticker.close
        }
    except Exception as e:
        logger.exception(f"Error getting market data for {symbol}")
        raise HTTPException(
            status_code=400, 
            detail=str(e) if not DEBUG else f"{str(e)}\n\n{traceback.format_exc()}"
        )

@app.get("/account")
async def how_much():
    try:
        return await get_account_summary(ib_client.ib)
    except Exception as e:
        logger.exception("Error in account endpoint")
        raise HTTPException(
            status_code=500, 
            detail=str(e) if not DEBUG else f"{str(e)}\n\n{traceback.format_exc()}"
        )
    
async def get_account_summary(ib: IB) -> dict:
    if not ib.isConnected():
        raise HTTPException(status_code=503, detail="Not connected to IB Gateway")
    
    try:
        await ib.reqAccountSummaryAsync()
        
        # Then grab the actual values from the wrapper
        # The data should now be in ib.accountSummary()
        account_values = ib.accountSummary()
        
        acct = {}
        for value in account_values:
            if value.tag == 'AvailableFunds':
                acct['AvailableFunds'] = float(value.value)
            if value.tag == "NetLiquidation":
                acct['NetLiquidation'] = float(value.value)
            if value.tag == "TotalCashValue":
                acct['TotalCashValue'] = float(value.value)
            if value.tag == "BuyingPower":
                acct['BuyingPower'] = float(value.value)
            if value.tag == "GrossPositionValue":
                acct['GrossPositionValue'] = float(value.value)
            if value.tag == "InitMarginReq":
                acct['InitMarginReq'] = float(value.value)
            if value.tag == "MaintMarginReq":
                acct['MaintMarginReq'] = float(value.value)

        return acct
        
    except Exception as e:
        logger.exception(f"Error getting account summary")
        raise HTTPException(
            status_code=500, 
            detail=f"Error getting account summary: {str(e)}" if not DEBUG else f"{str(e)}\n\n{traceback.format_exc()}"
        )

async def calculate_shares_to_buy(ib: IB, symbol: str, cash_per_position: float) -> int:
    """Calculate number of shares to buy based on current market price"""
    try:
        contract = Stock(symbol, 'SMART', 'USD')
        tickers = await ib.reqTickersAsync(contract)
        
        if not tickers:
            error_msg = f"Could not get current price for {symbol}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Ticker[0] %s", tickers[0])
        logger.info("Ticker[0].marketPrice() %f", tickers[0].marketPrice())
        current_price = tickers[0].marketPrice()
        
        if isnan(current_price) or current_price <= 0:
            error_msg = f"Invalid or unavailable market price for {symbol}: {current_price}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Calculate shares, round down to nearest whole share
        shares = int(cash_per_position / current_price)
        return shares
    except Exception as e:
        logger.exception(f"Error calculating shares to buy for {symbol}")
        raise

async def wait_for_order(trade: Trade, timeout=30):
    """Wait for an order to complete with timeout"""
    try:
        start_time = time.time()
        while not trade.isDone():
            if time.time() - start_time > timeout:
                error_msg = f"Order did not complete within {timeout} seconds"
                logger.error(error_msg)
                raise TimeoutError(error_msg)
            # FIX: Use asyncio.sleep instead of ib_client.ib.sleep
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

async def validate_contract(ib_client, symbol: str) -> Stock:
    """Validate and qualify a stock contract"""
    try:
        contract = Stock(symbol, 'SMART', 'USD')
        qualified_contracts = await ib_client.ib.qualifyContractsAsync(contract)
        if not qualified_contracts:
            error_msg = f"Could not validate contract for {symbol}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        return qualified_contracts[0]
    except Exception as e:
        logger.exception(f"Error validating contract for {symbol}")
        raise

@app.get("/daily")
async def do_buys_for_day():
    # At the beginning of your do_buys_for_day function
    logger.warning(f"do_buys_for_day was called by: {traceback.format_stack()}")

    """Execute daily orders - sell non-leaders and prepare to buy new leaders"""
    if not ib_client.ib.isConnected():
        raise HTTPException(status_code=503, detail="Not connected to IB Gateway")

    try:
        # Get current leaders
        leaders = await get_positive_leaders(ib_client.ib)
        if not leaders:
            logger.warning("No leaders found for today")
            return {"positions_closed": [], "leaders_to_buy": [], "buys_executed": []}

        # Get current portfolio positions
        portfolio = await ib_client.ib.reqPositionsAsync()
        
        # Track actions taken
        actions = {
            "positions_closed": [],
            "leaders_to_buy": leaders,
            "buys_executed": []
        }

        # Get initial cash before any trades
        initial_cash = await get_account_summary(ib_client.ib)
        if initial_cash["AvailableFunds"] <= 0:
            raise HTTPException(status_code=400, detail="Insufficient funds in account")

        # Process each position
        if portfolio:
            for position in portfolio:
                symbol = position.contract.symbol
                if symbol not in leaders:
                    try:
                        # Validate contract first
                        contract = await validate_contract(ib_client, symbol)
                        quantity = abs(position.position)
                        
                        # Create opposite order (sell if long, buy if short)
                        side = 'SELL' if position.position > 0 else 'BUY'
                        order = MarketOrder(side, quantity)
                        
                        # Submit the order
                        trade = ib_client.ib.placeOrder(contract, order)

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
        acct_summary = await get_account_summary(ib_client.ib)
        available_cash = acct_summary["AvailableFunds"]
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
                # Validate contract
                contract = await validate_contract(ib_client, symbol)
                
                # Calculate shares to buy
                shares = await calculate_shares_to_buy(ib_client.ib, symbol, cash_per_position)
                
                if shares <= 0:
                    logger.warning(f"Insufficient funds to buy {symbol} at current price")
                    continue
                
                logger.info("Buying %d %s", shares, symbol)
                
                # Create and place buy order
                order = MarketOrder('BUY', shares)
                trade = ib_client.ib.placeOrder(contract, order)
                
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

# Remove the old start_server function and modify __main__ block
if __name__ == '__main__':
    try:
        port = int(os.getenv('PORT', '8080'))
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
    except Exception as e:
        print(f'Fatal error: {e}')
        print(f'Traceback: {traceback.format_exc()}')
        # No need to call ib_client.stop() as the lifespan manager handles this