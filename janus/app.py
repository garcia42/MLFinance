from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from ib_insync import IB, Stock, Contract, MarketOrder
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

from datetime import datetime
from pytz import timezone
import time
import asyncio
from typing import Dict, List
from fastapi import HTTPException
from ib_insync import Stock, MarketOrder
from math import isnan

nest_asyncio.apply()

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        finally:
            sys.exit(0)

    async def connect_ib(self):
        """Async function to connect to IB Gateway"""
        try:
            await self.ib.connectAsync(**self.ib_config)
            print('Successfully connected to IB Gateway')
        except Exception as e:
            print(f'Connection error: {e}')
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
            finally:
                print("Event loop stopped. Restarting in 2 seconds...")
                asyncio.sleep(2)

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

# Create global IB client instance
ib_client = IBClient()

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
                        'error': str(e)
                    }
        except Exception as e:
            status['error'] = str(e)
    
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
        raise HTTPException(status_code=500, detail=str(e))

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
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/account")
async def how_much():
    return await get_account_summary(ib_client.ib)
    
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
        print(f"Error getting account summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting account summary: {str(e)}")

async def calculate_shares_to_buy(ib: IB, symbol: str, cash_per_position: float) -> int:
    """Calculate number of shares to buy based on current market price"""
    contract = Stock(symbol, 'SMART', 'USD')
    tickers = await ib.reqTickersAsync(contract)
    
    if not tickers:
        raise ValueError(f"Could not get current price for {symbol}")
    
    logger.info("Ticker[0] %s", tickers[0])
    logger.info("Ticker[0].marketPrice() %f", tickers[0].marketPrice())  # Changed %d to %f
    current_price = tickers[0].marketPrice()
    if isnan(current_price) or current_price <= 0:  # Add nan check
        raise ValueError(f"Invalid or unavailable market price for {symbol}: {current_price}")

        
    # Calculate shares, round down to nearest whole share
    shares = int(cash_per_position / current_price)
    return shares

async def wait_for_order(trade, timeout=30):
    """Wait for an order to complete with timeout"""
    start_time = time.time()
    while not trade.isDone():
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Order did not complete within {timeout} seconds")
        await ib_client.ib.sleep(1)

def is_market_hours(current_time: datetime) -> bool:
    """Check if we're in regular market hours (9:30 AM - 4:00 PM ET, Mon-Fri)"""
    if current_time.weekday() > 4:  # Saturday = 5, Sunday = 6
        return False
    market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= current_time <= market_close

async def validate_contract(ib_client, symbol: str) -> Stock:
    """Validate and qualify a stock contract"""
    contract = Stock(symbol, 'SMART', 'USD')
    qualified_contracts = await ib_client.ib.qualifyContractsAsync(contract)
    if not qualified_contracts:
        raise ValueError(f"Could not validate contract for {symbol}")
    return qualified_contracts[0]

@app.get("/daily")
async def do_buys_for_day():
    """Execute daily orders - sell non-leaders and prepare to buy new leaders"""
    if not ib_client.ib.isConnected():
        raise HTTPException(status_code=503, detail="Not connected to IB Gateway")

    # # Check market hours
    # current_time = datetime.now(timezone('US/Eastern'))
    # if not is_market_hours(current_time):
    #     raise HTTPException(status_code=400, detail="Market is currently closed")

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
                        logger.error(f"Failed to close position for {symbol}: {e}")
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
                
                # Wait for order with timeout
                try:
                    await wait_for_order(trade)
                except TimeoutError as e:
                    logger.error(f"Order timeout for {symbol}: {e}")
                    continue

                # Check order status
                if trade.orderStatus.status != "Filled":
                    raise Exception(f"Order failed with status: {trade.orderStatus.status}")
                
                actions["buys_executed"].append({
                    "symbol": symbol,
                    "shares": shares,
                    "filled_shares": trade.orderStatus.filled,
                    "status": trade.orderStatus.status,
                    "avg_fill_price": trade.orderStatus.avgFillPrice
                })
                
                logger.info(f"Bought {trade.orderStatus.filled} shares of {symbol} at {trade.orderStatus.avgFillPrice}")
                
            except Exception as e:
                logger.error(f"Error buying {symbol}: {e}")
                actions["buys_executed"].append({
                    "symbol": symbol,
                    "error": str(e)
                })
        
        return actions
        
    except Exception as e:
        logger.error(f"Error in daily execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
def start_server():
    try:
        ib_client.start()
        port = int(os.getenv('PORT', '8080'))
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
    except Exception as e:
        print(f'Fatal error: {e}')
        ib_client.stop()

if __name__ == '__main__':
    start_server()