# Standard library modules
import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple

# Third-party modules
import fredapi
import nest_asyncio
import numpy as np
import pandas as pd
import praw
import pytz
import tweepy
from dataclasses import dataclass
from ib_insync import *

# FinancialMachineLearning modules
from FinancialMachineLearning.features.fracdiff import FractionalDifferentiatedFeatures

# Claude modules
from claude.contract_util import get_next_quarterly_expiry
from claude.feature_storage import FeatureStorage
from claude.fred_collector import FredDataCollector
from claude.historic_data_collector import create_data_collector
from claude.runbar import RunBarsProcessor
from claude.train_model import label_and_analyze
from claude.validate import label_and_cross_validate, calculate_psr, build_model


# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()


@dataclass
class OrderFlowMetrics:
    """Container for order flow analysis metrics"""
    aggressive_buy_volume: float = 0.0
    aggressive_sell_volume: float = 0.0
    buy_sell_imbalance: float = 0.0
    large_order_count: int = 0
    trade_count: int = 0

class VolumeProfile:
    """Volume Profile analysis"""
    def __init__(self, price_data: List[float], volume_data: List[float], num_bins: int = 50):
        self.prices = np.array(price_data)
        self.volumes = np.array(volume_data)
        self.num_bins = num_bins
        self.profile = self._calculate_profile()
        
    def _calculate_profile(self) -> Dict[float, float]:
        """Calculate volume distribution across price levels"""
        if len(self.prices) == 0:
            return {}
            
        bins = np.linspace(min(self.prices), max(self.prices), self.num_bins)
        hist, bin_edges = np.histogram(self.prices, bins=bins, weights=self.volumes)
        return dict(zip(bin_edges[:-1], hist))
        
    def get_value_areas(self) -> Tuple[float, float, float]:
        """Calculate value area high, low, and point of control"""
        if not self.profile:
            return 0.0, 0.0, 0.0
            
        total_volume = sum(self.profile.values())
        poc_price = max(self.profile.items(), key=lambda x: x[1])[0]
        
        # Calculate 70% value area
        cumsum = 0
        sorted_prices = sorted(self.profile.items(), key=lambda x: x[1], reverse=True)
        value_area_prices = []
        
        for price, volume in sorted_prices:
            cumsum += volume
            value_area_prices.append(price)
            if cumsum >= 0.7 * total_volume:
                break
                
        return min(value_area_prices), max(value_area_prices), poc_price

class MarketMicrostructure:
    """Analysis of market microstructure metrics"""
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.bid_history = []
        self.ask_history = []
        self.trade_history = []
        
    def update(self, bid: float, ask: float, last_price: float, volume: float):
        """Update market state with new data"""
        self.bid_history.append(bid)
        self.ask_history.append(ask)
        self.trade_history.append((last_price, volume))
        
        # Keep only recent history
        if len(self.bid_history) > self.window_size:
            self.bid_history.pop(0)
            self.ask_history.pop(0)
            self.trade_history.pop(0)
            
    def get_metrics(self) -> Dict[str, float]:
        """Calculate market microstructure metrics"""
        if not self.bid_history or not self.ask_history:
            return {
                'spread': 0.0,
                'spread_volatility': 0.0,
                'quote_stability': 0.0,
                'price_impact': 0.0
            }
            
        # Calculate basic spread metrics
        spreads = [ask - bid for bid, ask in zip(self.bid_history, self.ask_history)]
        current_spread = spreads[-1]
        spread_volatility = np.std(spreads) if len(spreads) > 1 else 0.0
        
        # Quote stability (lower means more stable)
        quote_changes = sum(1 for i in range(1, len(self.bid_history))
                          if self.bid_history[i] != self.bid_history[i-1]
                          or self.ask_history[i] != self.ask_history[i-1])
        quote_stability = quote_changes / len(self.bid_history) if self.bid_history else 0.0
        
        # Approximate price impact using volume-weighted price changes
        price_impacts = []
        for i in range(1, len(self.trade_history)):
            price_change = abs(self.trade_history[i][0] - self.trade_history[i-1][0])
            volume = self.trade_history[i][1]
            if volume > 0:
                price_impacts.append(price_change / volume)
        
        avg_price_impact = np.mean(price_impacts) if price_impacts else 0.0
        
        return {
            'spread': current_spread,
            'spread_volatility': spread_volatility,
            'quote_stability': quote_stability,
            'price_impact': avg_price_impact
        }

class IntegratedMarketCollector:
    def __init__(self,
                 reddit_client_id: str,
                 reddit_client_secret: str,
                 reddit_user_agent: str,
                 fred_api_key: str,
                 twitter_bearer_token: str = None):
        """Initialize all API connections and analysis components"""
        # Initialize IB
        self.ib = IB()
        
        # Initialize Reddit with rate limiting
        self.reddit = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent=reddit_user_agent
        )
        
        # Initialize Twitter if token provided
        self.twitter_client = (tweepy.Client(bearer_token=twitter_bearer_token) 
                             if twitter_bearer_token else None)
        
        # Initialize FRED
        self.fred = fredapi.Fred(api_key=fred_api_key)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize data storage
        self.market_data = []
        self.macro_data = None
        
        # Initialize analysis components
        self.volume_profile = None
        self.microstructure = MarketMicrostructure()
        self.order_flow = OrderFlowMetrics()
        
        # Trading hours timezone
        self.timezone = pytz.timezone('US/Eastern')
        
        # Rate limiting settings
        self.reddit_calls = 0
        self.twitter_calls = 0
        self.last_reddit_reset = time.time()
        self.last_twitter_reset = time.time()
        
        # FRED series to track
        self.fred_series = FredDataCollector.FRED_SERIES
        
    def create_es_contract(self):
        """Create an ES futures contract object"""
        contract = Future('ES', exchange='CME', lastTradeDateOrContractMonth=get_next_quarterly_expiry())
        
        all_contracts = self.ib.qualifyContracts(contract)

        # Get front-month contract
        print(all_contracts[0])
        return all_contracts[0] if all_contracts else None
                
    def _is_market_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is during regular ES futures trading hours"""
        ts_eastern = timestamp.astimezone(self.timezone)
        
        if ts_eastern.weekday() == 5:  # Saturday
            return False
        elif ts_eastern.weekday() == 6:  # Sunday
            return ts_eastern.hour >= 18
        elif ts_eastern.weekday() == 4:  # Friday
            return ts_eastern.hour < 17
        else:
            return True
        
    def update_order_flow_metrics(self, trade: Trade):
        """Update order flow metrics based on new trade"""
        if trade.price >= trade.ask:  # Aggressive buy
            self.order_flow.aggressive_buy_volume += trade.volume
        elif trade.price <= trade.bid:  # Aggressive sell
            self.order_flow.aggressive_sell_volume += trade.volume
            
        total_volume = self.order_flow.aggressive_buy_volume + self.order_flow.aggressive_sell_volume
        if total_volume > 0:
            self.order_flow.buy_sell_imbalance = (
                (self.order_flow.aggressive_buy_volume - self.order_flow.aggressive_sell_volume)
                / total_volume
            )
            
        # Track large orders (> 10 contracts for ES)
        if trade.volume > 10:
            self.order_flow.large_order_count += 1
        self.order_flow.trade_count += 1
            
    def update_macro_data(self):
        """Update macro economic data from FRED with error handling"""
        macro_data = {}
        
        for name, series_id in self.fred_series.items():
            for attempt in range(3):  # Retry up to 3 times
                try:
                    series = self.fred.get_series(series_id)
                    df = pd.DataFrame(series)
                    df.columns = [name]
                    macro_data[name] = df
                    self.logger.info(f"Updated {name} data from FRED")
                    break
                except Exception as e:
                    if attempt == 2:  # Last attempt
                        self.logger.error(f"Failed to fetch {name} from FRED after 3 attempts: {e}")
                    else:
                        self.logger.warning(f"Retry {attempt + 1} for {name} from FRED")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        
        if macro_data:
            self.macro_data = pd.concat(macro_data.values(), axis=1)
            self.macro_data.index = pd.to_datetime(self.macro_data.index)
            self._calculate_macro_indicators()
            
    def _calculate_macro_indicators(self):
        """Calculate additional macro indicators with error handling"""
        if self.macro_data is None:
            return
            
        try:
            df = self.macro_data
            
            # YoY changes
            for col in df.columns:
                df[f'{col}_YOY'] = df[col].pct_change(periods=12)
                
            # Economic regime indicators
            df['recession_probability'] = (
                (df['UNRATE_YOY'] > 0) &
                (df['INDPRO_YOY'] < 0) &
                (df['T10Y2Y'] < 0)
            ).astype(float)
            
            # Inflation regime
            df['high_inflation'] = (df['CPIAUCSL_YOY'] > 0.03).astype(float)
            
            # Monetary conditions
            df['tight_money'] = (
                (df['FEDFUNDS'].diff() > 0) &
                (df['M2_YOY'] < df['M2_YOY'].rolling(12).mean())
            ).astype(float)
            
            self.macro_data = df
            
        except Exception as e:
            self.logger.error(f"Error calculating macro indicators: {e}")
            
    def get_macro_context(self) -> Dict:
        """Get current macro context with error handling"""
        if self.macro_data is None or self.macro_data.empty:
            return {}
            
        try:
            latest = self.macro_data.iloc[-1]
            
            return {
                'fed_rate': latest['FEDFUNDS'],
                'unemployment': latest['UNRATE'],
                'inflation_yoy': latest['CPIAUCSL_YOY'],
                'yield_spread': latest['T10Y2Y'],
                'recession_prob': latest['recession_probability'],
                'monetary_regime': 'tight' if latest['tight_money'] else 'loose',
                'inflation_regime': 'high' if latest['high_inflation'] else 'low'
            }
        except Exception as e:
            self.logger.error(f"Error getting macro context: {e}")
            return {}
            
    def create_feature_matrix(self) -> pd.DataFrame:
        """Combine market, sentiment, and macro data into feature matrix"""
        try:
            # Convert market data to DataFrame
            market_df = pd.DataFrame(self.market_data)
            market_df.set_index('timestamp', inplace=True)
            
            # Create run bars
            run_bars_processor = RunBarsProcessor(
                threshold_type='dollar',
                threshold=1_000_000
            )
            
            market_bars = run_bars_processor.process_market_data(market_df)
            
            # Update market data with run bars
            market_df = market_bars
            
            # Resample market data to 5-minute bars
            market_bars = market_df.resample('5T').agg({
                'price': ['first', 'high', 'low', 'last'],
                'volume': 'sum',
                'bid': 'last',
                'ask': 'last',
                'spread': 'mean',
                'spread_volatility': 'last',
                'quote_stability': 'last',
                'price_impact': 'mean',
                'buy_sell_imbalance': 'last',
                'large_order_ratio': 'last',
                'value_area_high': 'last',
                'value_area_low': 'last',
                'point_of_control': 'last',
                'fed_rate': 'last',
                'unemployment': 'last',
                'inflation_yoy': 'last',
                'yield_spread': 'last',
                'recession_prob': 'last'
            })
            
            # Combine data
            features = pd.concat([market_bars], axis=1)
            
            # Calculate market features
            features['returns'] = features['price']['last'].pct_change()
            features['volatility'] = features['returns'].rolling(12).std()
            features['volume_delta'] = features['volume'].diff()
            
            # Calculate additional technical features
            features['rsi'] = self._calculate_rsi(features['price']['last'])
            features['macd'], features['macd_signal'] = self._calculate_macd(features['price']['last'])
            
            # Calculate regime-based features
            features['high_volatility_regime'] = (
                features['volatility'] > features['volatility'].rolling(24).mean()
            ).astype(int)
            
            features['risk_regime'] = np.select(
                [
                    (features['recession_prob'] > 0.7) & (features['volatility'] > features['volatility'].rolling(24).mean()),
                    (features['recession_prob'] > 0.7) & (features['volatility'] <= features['volatility'].rolling(24).mean()),
                    (features['recession_prob'] <= 0.7) & (features['volatility'] > features['volatility'].rolling(24).mean())
                ],
                ['high_risk', 'recession_risk', 'vol_risk'],
                default='normal'
            )
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating feature matrix: {e}")
            raise
            
    def _calculate_rsi(self, prices: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return pd.Series(index=prices.index)
            
    def _calculate_macd(self, prices: pd.Series,
                       fast: int = 12,
                       slow: int = 26,
                       signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and Signal line"""
        try:
            exp1 = prices.ewm(span=fast, adjust=False).mean()
            exp2 = prices.ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            return macd, signal_line
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return pd.Series(index=prices.index), pd.Series(index=prices.index)
            
    async def run_data_collection(self, duration_hours: float = 6.5):
        """Run the complete data collection process"""
        try:
            # Connect to IB
            self.ib.connect(host='127.0.0.1', port=7497, clientId=1)
            
            # Create ES contract
            contract = self.create_es_contract()

            storage = FeatureStorage('./Data/financial_features.parquet')
    
            historic_data = storage.load_existing_features()[0]
            # return historic_data

            # Define your desired date range
            full_start_date = datetime(2015, 1, 1)

            end_date = datetime.now()
            
            # Get the date range we need to fetch
            latest_stored_date, _ = storage.get_missing_dates(full_start_date, end_date)
            
            print(f"latest_stored_date: {latest_stored_date}")
            
            # Fetch the historical data since the last fetch, add 1 day so that the macro data can be ahead of micro
            collector = create_data_collector(ib_connection=self.ib, fred_api=self.fred, reddit=self.reddit, contract=contract, lookback_days=(end_date - latest_stored_date).days + 1)
            # Collect data
            historic_data = await collector.collect_all_historical_data(contract)

            print("Data shape:", historic_data.shape)
            print("\nColumns:", historic_data.columns.tolist())
            
            # Update storage with new features
            updated_features = storage.update_features(historic_data)

            print("HISTORIC IN CLAUDE")
            print(updated_features)

            # TODO first do historical
            # Collect data
            # duration_seconds = int(duration_hours * 3600)
            # await self.collect_market_data(contract, duration_seconds)

            # Create and save feature matrix
            # features = self.create_feature_matrix()
            
            frac_diff = FractionalDifferentiatedFeatures.fracDiff(updated_features[['close']], .2)
            
            updated_features['close'] = frac_diff

            return updated_features
            
        except Exception as e:
            self.logger.error(f"Error in data collection: {e}")
            raise
        finally:
            self.ib.disconnect()

def main():
    # Initialize collector with error handling
    try:
        collector = IntegratedMarketCollector(
            reddit_client_id='vaANhmFSKa7HMntggd324A',
            reddit_client_secret='6-gPDtvtoqKLaS9_19hm_3faAhB1IA',
            reddit_user_agent='collector',
            fred_api_key='c44011f35ea9b58dc265ab237efaa525',
            twitter_bearer_token='AAAAAAAAAAAAAAAAAAAAAGDQwgEAAAAAAn2onqKyAxz7MIQHabh8kRTnXdI%3DFlVh9CHWs57VIhAyF29lCX9FAkrFyyA8mFOns3scoKtAqmlfLy'  # Optional
        )
        
        async def run():
            return await collector.run_data_collection(duration_hours=6.5)
        
        loop = asyncio.get_event_loop()
        features = loop.run_until_complete(run())
        
        return features
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == '__main__':
    features_matrix = main()
    model = build_model(features=features_matrix, use_cache=False)
    # psr = calculate_psr(model)
    # print("PSR:", psr)
    # label_and_analyze(features=None)
    # label_and_analyze(features=features_matrix)
    label_and_cross_validate(features=features_matrix)
    