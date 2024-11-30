import asyncio
import json
import websockets
from datetime import datetime

class PolygonWebSocketClient:
    def __init__(self, api_key, symbols):
        """
        Initialize the WebSocket client
        
        Args:
            api_key (str): Your Polygon.io API key
            symbols (list): List of stock symbols to subscribe to
        """
        self.api_key = api_key
        self.symbols = [sym.upper() for sym in symbols]
        self.ws_url = "wss://delayed.polygon.io/stocks"
        self.connection = None
        
    async def connect(self):
        """Establish WebSocket connection and authenticate"""
        self.connection = await websockets.connect(self.ws_url)
        auth_message = {"action": "auth", "params": self.api_key}
        await self.connection.send(json.dumps(auth_message))
        
        # Wait for authentication response
        response = await self.connection.recv()
        if "connected" not in response:
            raise Exception("Authentication failed")
            
    async def subscribe(self):
        """Subscribe to specified stock symbols"""
        subscribe_message = {
            "action": "subscribe",
            "params": [f"T.{symbol}" for symbol in self.symbols]  # T.* for trades
        }
        await self.connection.send(json.dumps(subscribe_message))
        
    async def handle_message(self, message):
        """
        Process incoming WebSocket messages
        Override this method to implement custom handling
        """
        data = json.loads(message)
        if data[0]['ev']:  # Trade event
            for trade in data:
                print(trade)
                # timestamp = datetime.fromtimestamp(trade['t']/1000)
                # print(f"Trade: {trade['sym']} - Price: ${trade['p']:.2f}, "
                #       f"Size: {trade['s']}, Time: {timestamp}")

    async def listen(self):
        """Main loop to receive messages"""
        try:
            while True:
                message = await self.connection.recv()
                await self.handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed. Attempting to reconnect...")
            await self.start()
            
    async def start(self):
        """Start the WebSocket client"""
        await self.connect()
        await self.subscribe()
        await self.listen()
        
    def run(self):
        """Run the client in the event loop"""
        asyncio.get_event_loop().run_until_complete(self.start())

# Example usage
if __name__ == "__main__":
    API_KEY = "9Nvwv62Oh4mdiGBtjDvl5p3thVE2goB9"
    SYMBOLS = ["AAPL", "MSFT", "GOOGL"]
    
    client = PolygonWebSocketClient(API_KEY, SYMBOLS)
    client.run()