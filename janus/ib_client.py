import threading
import os
from concurrent.futures import ThreadPoolExecutor
from ib_insync import IB, Stock, Contract, MarketOrder, Trade, ContFuture, util, ExecutionFilter
import signal
import asyncio
import traceback
import sys
import logging
import functools
import time
import pandas as pd

from fastapi import HTTPException

logger = logging.getLogger(__name__)

class IBClient:
    def __init__(self):
        self.ib = IB()
        self.ib_config = {
            'host': os.getenv('IB_HOST', '127.0.0.1'),
            # https://github.com/gnzsnz/ib-gateway-docker, 
            'port': int(os.getenv('IB_PORT', '4004')),  # 4003 for live, 4004 for paper
            'clientId': int(os.getenv('IB_CLIENT_ID', '1')),
        }
        self.connection_lock = threading.Lock()
        self.connected = False
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        self.shutdown_event = asyncio.Event()

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f'Received signal {signum}. Starting shutdown...')
        try:
            if self.ib.isConnected():
                # Just disconnect synchronously during shutdown
                self.ib.disconnect()
                print("IB Gateway disconnected")
            
            # Shutdown the executor
            self.executor.shutdown(wait=False)
            print("Thread executor shutdown")
            
            # Set the shutdown event to notify any waiting coroutines
            if not self.shutdown_event.is_set():
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.call_soon_threadsafe(self.shutdown_event.set)
                else:
                    self.shutdown_event.set()
        except Exception as e:
            print(f'Error during shutdown: {e}')
            print(f'Traceback: {traceback.format_exc()}')
        finally:
            # Do not call sys.exit() here - let the main application handle this
            pass

    async def ensure_connected(self):
        """Ensure IB connection is active"""
        with self.connection_lock:
            if not self.ib.isConnected():
                try:
                    logger.info("Connecting to IB Gateway...")
                    await self.ib.connectAsync(
                        host=self.ib_config['host'],
                        port=self.ib_config['port'],
                        clientId=self.ib_config['clientId']
                    )
                    self.connected = True
                    logger.info("Successfully connected to IB Gateway")
                except Exception as e:
                    self.connected = False
                    logger.error(f"Failed to connect to IB Gateway: {e}")
                    if 'port' in str(e).lower() or 'connection refused' in str(e).lower():
                        logger.critical('Critical connection error')
                        raise HTTPException(status_code=503, detail=f"Cannot connect to IB Gateway: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Error connecting to IB Gateway: {str(e)}")
            return self.ib
    
    async def run_in_executor(self, func, *args, **kwargs):
        """Run a synchronous function in the thread pool executor
        
        This is the key method that allows us to run synchronous IB operations
        in a separate thread, avoiding "event loop already running" errors.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: func(*args, **kwargs)
        )
    
    # Helper methods for common IB operations
    
    async def get_positions(self):
        """Get current positions using async approach"""
        # Prefer the async version if available
        try:
            return await self.ib.reqPositionsAsync()
        except Exception as e:
            logger.exception("Error getting positions asynchronously")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_market_data(self, contract):
        """Get market data for a contract using thread executor"""
        # Run the synchronous reqMktData in a separate thread
        try:
            # Ensure we have a qualified contract
            qualified_contracts = await self.ib.qualifyContractsAsync(contract)
            if not qualified_contracts:
                raise ValueError(f"Could not qualify contract: {contract.symbol}")
            
            # Run the potentially blocking operation in a separate thread
            def get_data():
                ticker = self.ib.reqMktData(qualified_contracts[0])
                self.ib.sleep(1)  # Allow some time for data to arrive
                return ticker
            
            return await self.run_in_executor(get_data)
        except Exception as e:
            logger.exception(f"Error getting market data for {contract.symbol}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def place_order(self, contract, order):
        """Place an order asynchronously"""
        try:
            # Use the async version for placing orders
            trade = self.ib.placeOrder(contract, order)
            return trade
        except Exception as e:
            logger.exception(f"Error placing order for {contract.symbol}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_account_summary(self):
        """Get account summary using thread executor for sync operations"""
        try:
            # First ensure we're connected
            await self.ensure_connected()
            
            account_values = await self.ib.accountSummaryAsync()
        
            logger.info("Account summary request completed")
            
            # Convert account values to a dictionary for easier use
            result = {}
            for val in account_values:
                if hasattr(val, 'tag') and hasattr(val, 'value'):
                    result[val.tag] = val.value
                else:
                    logger.warning(f"Unexpected account value format: {val}")
            
            logger.info(f"Processed {len(result)} account values")
            
            return result

        except Exception as e:
            logger.exception(f"Error getting account summary", e)
            raise
            
            

    async def calculate_shares_to_buy(self, symbol, cash_per_position):
        """Calculate number of shares to buy based on current market price"""
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            tickers = await self.ib.reqTickersAsync(contract)
            
            if not tickers:
                error_msg = f"Could not get current price for {symbol}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info("Ticker[0] %s", tickers[0])
            logger.info("Ticker[0].marketPrice() %f", tickers[0].marketPrice())
            current_price = tickers[0].marketPrice()
            
            if current_price <= 0:
                error_msg = f"Invalid or unavailable market price for {symbol}: {current_price}"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Calculate shares, round down to nearest whole share
            shares = int(cash_per_position / current_price)
            return shares
        except Exception as e:
            logger.exception(f"Error calculating shares to buy for {symbol}")
            raise
    
    async def validate_contract(self, symbol):
        """Validate and qualify a stock contract"""
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            qualified_contracts = await self.ib.qualifyContractsAsync(contract)
            if not qualified_contracts:
                error_msg = f"Could not validate contract for {symbol}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            return qualified_contracts[0]
        except Exception as e:
            logger.exception(f"Error validating contract for {symbol}")
            raise
    
    async def get_futures_data(self, symbol='GC', exchange='COMEX'):
        """Fetch daily gold futures data for analysis using continuous contract"""
        try:
            # Create a continuous futures contract
            # Use '@' symbol to indicate a continuous contract
            gold_contract = ContFuture(symbol=symbol, exchange=exchange)
            
            # No need to qualify a continuous contract
            
            # Request daily historical data (1 year)
            bars = await self.ib.reqHistoricalDataAsync(
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

    async def get_front_futures_contract(self, symbol, exchange, currency='USD') -> Contract:
        """
        Find the front futures contract based on trading volume.
        
        Args:
            symbol (str): The futures symbol (e.g., 'ES', 'NQ', 'CL')
            exchange (str): The exchange (e.g., 'CME', 'NYMEX')
            currency (str): Currency for the contract (default: 'USD')
            
        Returns:
            Contract: The front month futures contract with highest volume
        """
        try:
            await self.ensure_connected()
            
            # Create a generic futures contract to get all expirations
            contract = Contract()
            contract.symbol = symbol
            contract.secType = 'FUT'
            contract.exchange = exchange
            contract.currency = currency
            
            # Get matching contracts
            contracts = await self.ib.reqContractDetailsAsync(contract)
            
            if not contracts:
                error_msg = f"No futures contracts found for {symbol} on {exchange}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info(f"Found {len(contracts)} futures contracts for {symbol}")
            
            # Extract the actual contracts from contract details
            futures_contracts = [details.contract for details in contracts]
            
            # Request market data for each contract to get volume
            tickers = await self.ib.reqTickersAsync(*futures_contracts)
            
            # Create a list of (contract, volume) tuples
            contract_volumes = []
            for ticker, contract in zip(tickers, futures_contracts):
                volume = ticker.volume if hasattr(ticker, 'volume') and ticker.volume is not None else 0
                contract_volumes.append((contract, volume))
                logger.info(f"{contract.localSymbol}: Volume = {volume}")
            
            # Sort by volume (highest first)
            contract_volumes.sort(key=lambda x: x[1], reverse=True)
            
            # Return the contract with highest volume
            if not contract_volumes:
                error_msg = f"Could not get volume data for {symbol} futures"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            front_contract = contract_volumes[0][0]
            logger.info(f"Selected front contract {front_contract.localSymbol} with volume {contract_volumes[0][1]}")
            
            return front_contract
            
        except Exception as e:
            logger.exception(f"Error finding front futures contract for {symbol}")
            raise HTTPException(status_code=500, detail=str(e))

    def place_futures_trade(self, contract: Contract, symbol='GC',
                                        action='BUY', quantity=1) -> Trade:
        """Place a trade using the continuous futures contract"""
        try:
            # Create market order
            order = MarketOrder(action=action, totalQuantity=quantity)
            
            # Place the order
            trade = self.ib.placeOrder(contract, order)
            logger.info(f"Placed {action} order for {quantity} {symbol} continuous contract")
            
            return trade
            
        except Exception as e:
            logger.exception(f"Error placing continuous futures trade: {e}")
            return None
    
    async def get_gold_futures_trades(self):
        """
        Retrieves execution reports for gold futures contracts.
        Returns a list of trades in gold futures contracts matching GC or other gold symbols.
        """
        try:
            # Ensure connection
            await self.ensure_connected()
            
            # Create an empty execution filter
            exec_filter = ExecutionFilter()
            
            # Execute in thread pool to avoid blocking
            all_trades = await self.ib.reqExecutionsAsync(exec_filter)
            
            print(all_trades)
            
            # Filter for gold futures contracts
            gold_futures_trades = []
            
            for trade in all_trades:
                # Extract relevant information
                symbol = trade.execution.symbol
                sec_type = trade.contract.secType if hasattr(trade.contract, 'secType') else ''
                exchange = trade.execution.exchange if hasattr(trade.execution, 'exchange') else ''
                
                # Check if it's a futures contract related to gold
                if sec_type == 'FUT':
                    # Common gold futures symbols: GC (COMEX), YG (mini), ZG (micro), etc.
                    if (symbol in ['GC', 'YG', 'ZG', 'MGC'] or 
                        'GOLD' in symbol.upper() or 
                        (exchange == 'COMEX' and symbol == 'GC')):
                        gold_futures_trades.append(trade)
                    # Also check underlying if available
                    elif hasattr(trade.contract, 'underlying') and trade.contract.underlying in ['GOLD', 'XAU']:
                        gold_futures_trades.append(trade)
            
            logger.info(f"Found {len(gold_futures_trades)} gold futures trades")
            return gold_futures_trades
            
        except Exception as e:
            logger.exception(f"Error retrieving gold futures trades: {e}")
            raise HTTPException(status_code=500, detail=f"Error retrieving gold futures trades: {str(e)}")