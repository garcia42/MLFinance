from flask import Flask, jsonify
from ib_insync import IB, Stock, Contract
import threading
import os
import signal
import sys
from gunicorn.app.base import BaseApplication
from gcp import logger as logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

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
        logging.info(f'Received signal {signum}. Starting shutdown...')
        self.running = False
        try:
            if self.ib.isConnected():
                asyncio.run_coroutine_threadsafe(self.ib.disconnectAsync(), self.loop)
                self.executor.shutdown(wait=True)
        except Exception as e:
            logging.error(f'Error during shutdown: {e}')
        finally:
            sys.exit(0)

    async def connect_ib(self):
        """Async function to connect to IB Gateway"""
        try:
            await self.ib.connectAsync(**self.ib_config)
            logging.info('Successfully connected to IB Gateway')
        except Exception as e:
            logging.error(f'Connection error: {e}')
            # In Cloud Run, we might want to exit if we can't connect
            if 'port' in str(e).lower() or 'connection refused' in str(e).lower():
                logging.error('Critical connection error, exiting...')
                sys.exit(1)

    async def maintain_connection(self):
        """Async function to maintain IB connection"""
        while self.running:
            if not self.ib.isConnected():
                logging.info('Connecting to IB Gateway...')
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
                logging.error(f'Event loop crashed: {e}')
            finally:
                logging.warning("Event loop stopped. Restarting in 2 seconds...")
                asyncio.sleep(2)  # Prevent immediate loop restarts in case of repeated failures


    def start(self):
        """Start the connection maintenance thread"""
        if self.loop and self.loop.is_running():
            logging.info("Event loop already running")
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
            asyncio.run(self.ib.disconnectAsync())
        self.executor.shutdown(wait=True)

# Create global IB client instance
ib_client = IBClient()

@app.route('/health')
def health_check():
    """Health check endpoint for Cloud Run"""
    return jsonify({'status': 'healthy'})

@app.route('/status')
def get_status():
    """Get the connection status"""
    return jsonify({
        'connected': ib_client.ib.isConnected(),
        'client_id': ib_client.ib_config['clientId'],
        'mode': 'paper' if ib_client.ib_config['port'] == 4002 else 'live'
    })

@app.route('/positions')
def get_positions():
    """Get current positions"""
    if not ib_client.ib.isConnected():
        return jsonify({'error': 'Not connected to IB Gateway'}), 503
    
    if ib_client.loop is None or not ib_client.loop.is_running():
        return jsonify({'error': 'Internal event loop error'}), 500
    
    future = asyncio.run_coroutine_threadsafe(
        ib_client.ib.reqPositionsAsync(),
        ib_client.loop
    )
    
    try:
        positions = future.result(timeout=5)  # 5 second timeout
        return jsonify([{
            'contract': {
                'symbol': pos.contract.symbol,
                'secType': pos.contract.secType,
                'exchange': pos.contract.exchange,
                'currency': pos.contract.currency
            },
            'position': pos.position,
            'avgCost': pos.avgCost
        } for pos in positions])
    except TimeoutError:
        return jsonify({'error': 'Request timed out'}), 504
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/account')
def get_account():
    """Get account information"""
    if not ib_client.ib.isConnected():
        return jsonify({'error': 'Not connected to IB Gateway'}), 503
    
    if ib_client.loop is None or not ib_client.loop.is_running():
        return jsonify({'error': 'Internal event loop error'}), 500
    
    future = asyncio.run_coroutine_threadsafe(
        ib_client.ib.accountSummaryAsync(),
        ib_client.loop
    )
    
    try:
        account = future.result(timeout=5)  # 5 second timeout
        return jsonify([{
            'tag': value.tag,
            'value': value.value,
            'currency': value.currency,
            'account': value.account
        } for value in account])
    except TimeoutError:
        return jsonify({'error': 'Request timed out'}), 504
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/market_data/<symbol>')
def get_market_data(symbol):
    """Get market data for a symbol"""
    if not ib_client.ib.isConnected():
        return jsonify({'error': 'Not connected to IB Gateway'}), 503
    
    if ib_client.loop is None or not ib_client.loop.is_running():
        return jsonify({'error': 'Internal event loop error'}), 500
    
    contract = Stock(symbol, 'SMART', 'USD')
    try:
        # Run both operations in sequence
        future1 = asyncio.run_coroutine_threadsafe(
            ib_client.ib.qualifyContractsAsync(contract),
            ib_client.loop
        )
        future1.result(timeout=5)  # Wait for contract qualification
        
        future2 = asyncio.run_coroutine_threadsafe(
            ib_client.ib.reqTickersAsync(contract),
            ib_client.loop
        )
        [ticker] = future2.result(timeout=5)
        
        return jsonify({
            'symbol': symbol,
            'last': ticker.last,
            'bid': ticker.bid,
            'ask': ticker.ask,
            'volume': ticker.volume,
            'close': ticker.close
        })
    except TimeoutError:
        return jsonify({'error': 'Request timed out'}), 504
    except Exception as e:
        return jsonify({'error': str(e)}), 400

class GunicornApplication(BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        for key, value in self.options.items():
            if key in self.cfg.settings and value is not None:
                self.cfg.set(key.lower(), value)

    def load(self):
        return self.application

if __name__ == '__main__':
    try:
        # Start IB client connection thread
        ib_client.start()
        
        # Configure Gunicorn
        options = {
            'bind': f"0.0.0.0:{os.getenv('PORT', '8080')}",
            'workers': 1,  # Single worker due to IB connection constraints
            'timeout': 0,  # Disable timeout for long-running connections
            'worker_class': 'gevent',  # Use gevent for async support
            'preload_app': True,
        }
        
        # Start Gunicorn server
        GunicornApplication(app, options).run()
    except Exception as e:
        logging.error(f'Fatal error: {e}')
        ib_client.stop()