import requests
import json
from datetime import datetime
import logging
import os
from typing import Optional, Dict, Any

class SchwabTrader:
    def __init__(self, api_key: str, api_secret: str, account_id: str):
        """
        Initialize the Schwab trading client
        
        Args:
            api_key (str): Your Schwab API key
            api_secret (str): Your Schwab API secret
            account_id (str): Your Schwab account ID
        """
        self.base_url = "https://api.schwab.com/v1/trading"  # Example URL
        self.api_key = api_key
        self.api_secret = api_secret
        self.account_id = account_id
        self.session = requests.Session()
        self._setup_logging()
        self._authenticate()

    def _setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='schwab_trading.log'
        )
        self.logger = logging.getLogger(__name__)

    def _authenticate(self):
        """Authenticate with Schwab API"""
        auth_url = f"{self.base_url}/oauth/token"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.api_secret
        }
        
        try:
            response = self.session.post(auth_url, headers=headers, data=data)
            response.raise_for_status()
            auth_data = response.json()
            self.session.headers.update({
                "Authorization": f"Bearer {auth_data['access_token']}"
            })
            self.logger.info("Successfully authenticated with Schwab API")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Authentication failed: {str(e)}")
            raise

    def place_market_order(
        self,
        symbol: str,
        quantity: int,
        side: str = "buy",
        order_type: str = "market"
    ) -> Dict[str, Any]:
        """
        Place a market order to buy or sell a stock
        
        Args:
            symbol (str): The stock symbol (e.g., 'AAPL')
            quantity (int): Number of shares to buy/sell
            side (str): 'buy' or 'sell'
            order_type (str): Type of order (default: 'market')
            
        Returns:
            Dict containing the order response
        """
        order_url = f"{self.base_url}/orders"
        
        order_data = {
            "accountId": self.account_id,
            "symbol": symbol,
            "quantity": quantity,
            "side": side.upper(),
            "type": order_type.upper(),
            "timeInForce": "DAY"
        }

        try:
            response = self.session.post(
                order_url,
                json=order_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            order_response = response.json()
            
            self.logger.info(
                f"Successfully placed {side} order for {quantity} shares of {symbol}"
            )
            return order_response
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Order placement failed: {str(e)}")
            raise

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Check the status of an order
        
        Args:
            order_id (str): The ID of the order to check
            
        Returns:
            Dict containing the order status
        """
        status_url = f"{self.base_url}/orders/{order_id}"
        
        try:
            response = self.session.get(status_url)
            response.raise_for_status()
            status_response = response.json()
            
            self.logger.info(f"Retrieved status for order {order_id}")
            return status_response
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get order status: {str(e)}")
            raise

def main():
    # Load credentials from environment variables
    api_key = os.getenv("SCHWAB_API_KEY")
    api_secret = os.getenv("SCHWAB_API_SECRET")
    account_id = os.getenv("SCHWAB_ACCOUNT_ID")
    
    if not all([api_key, api_secret, account_id]):
        raise ValueError("Missing required environment variables")
    
    # Initialize trader
    trader = SchwabTrader(api_key, api_secret, account_id)
    
    # Example: Buy 10 shares of Apple stock
    try:
        order = trader.place_market_order(
            symbol="AAPL",
            quantity=10,
            side="buy"
        )
        print(f"Order placed successfully: {order}")
        
        # Check order status
        status = trader.get_order_status(order['orderId'])
        print(f"Order status: {status}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()``