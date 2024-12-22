'''
SecurityData holds info around SecurityData
'''

from typing import Optional, Dict, List, Tuple
import math

from autotrader.autoplot import AutoPlot
import pandas as pd
from finta import TA
from convert_ohlc import convert_data_weekly, convert_data_daily
from models.purchase import Purchase

class SecurityData():
    '''
    '''

    def __init__(self, ticker: str, hourly_data: pd.DataFrame, daily_macd: pd.DataFrame, weekly_ema: pd.Series, divergences: pd.DataFrame, daily_data: pd.DataFrame):
        self.ticker = ticker
        self.hourly_data = hourly_data
        self.daily_macd = daily_macd
        self.weekly_ema = weekly_ema
        self.divergences = divergences
        self.daily_data = daily_data
        self.last_price = hourly_data["Close"].iloc[-1]
        self.weekly_data = convert_data_weekly(self.hourly_data)
        self.week_impulse = self.impulse_system(self.weekly_data)
        self.day_impulse = self.impulse_system(self.daily_data)
        self.daily_season = self.calculate_season(self.daily_macd)
        self.daily_bb = TA.BBANDS(self.daily_data)
        self.dmi = TA.DMI(self.weekly_data)
        self.adx = TA.ADX(self.weekly_data)

    def get_stop_value(self, price: float, atr_count: float, is_long: bool):
        if is_long:
            return price - (atr_count * TA.ATR(self.hourly_data).iloc[-1])
        else:
            return price + (atr_count * TA.ATR(self.hourly_data).iloc[-1])
    
    def impulse_system(self, df: pd.DataFrame):
        macd = TA.MACD(df)
        macd["HISTOGRAM"] = macd["MACD"] - macd["SIGNAL"]
        ema = TA.EMA(df)

        bigger_macd = macd["HISTOGRAM"].iloc[-1] > macd["HISTOGRAM"].iloc[-2]
        bigger_ema = ema.iloc[-1] > ema.iloc[-2]

        return "Green" if bigger_ema and bigger_macd else "Red" if not bigger_ema and not bigger_macd else "Blue"

    def should_sell_stock(self, purchases: Dict[str, List[Purchase]]) -> bool:
        '''
        #TODO
        This could get stuck in a trading range and never sell
        need to check if it's in a trading range and sell
        '''
        p_list: List[Purchase] = purchases.get(self.ticker, None)
        if p_list is None or len(p_list) == 0:
            return False
        p = p_list[0]
        
        bb_width = self.daily_bb["BB_UPPER"] - self.daily_bb["BB_LOWER"]

        long_diff = self.last_price - p.buyPrice
        short_diff = p.buyPrice - self.last_price
        if p.isLong and (long_diff > bb_width.iloc[-1] * .35):
            # Sell if capturing 30% of the channel height)
            return True
        elif not p.isLong and (short_diff > bb_width.iloc[-1] * .35):
            # Sell if capturing 30% of the channel height)
            return True

        bb_width_ind = (self.daily_bb["BB_UPPER"] - self.daily_bb["BB_LOWER"]) / self.daily_bb["BB_MIDDLE"]
        
        # Also sell if the last 7 days have all had no volatility
        return bb_width_ind.tail(7).all() < 8
    
    def is_in_buy_value_town(self, is_long: bool) -> bool:
        '''
        True if the last price is below/above the value zone for long or short trades
        '''
        ema_12 = TA.EMA(self.daily_data, 12)
        ema_26 = TA.EMA(self.daily_data, 26)
        lowest = min(ema_12.iloc[-1], ema_26.iloc[-1])
        highest = max(ema_12.iloc[-1], ema_26.iloc[-1])
        
        return (is_long and self.last_price < lowest) or (not is_long and self.last_price > highest)

    def should_buy_stock(self, divergence_age: int, single_trade_max_loss: float, atr_count: float, cash: float) -> Optional[Purchase]:
        
        is_dmi_up = (self.dmi["DI+"].tail(6) > self.dmi["DI-"].tail(6)).all()
        if self.adx.iloc[-1] < 25:
            return None

        buy_time = self.hourly_data["t"].iloc[-1]
        if is_dmi_up and self.day_impulse != "Red" and self.week_impulse != "Red" and self.daily_season == "Spring" and self.is_in_buy_value_town(True): # LONG
            for i in range(len(self.divergences) - 1, len(self.divergences) - divergence_age, -1):
                if self.divergences["regularBull"][i] == 1.0:
                    stop = self.get_stop_value(price=self.last_price, is_long=True, atr_count=atr_count)
                    max_stock_count = math.floor(single_trade_max_loss / (self.last_price - stop))
                    stock_count = min(math.floor(cash/self.last_price), max_stock_count)

                    print("==================")
                    print(f"Hit Bullish Stock: {self.ticker}, buy {stock_count} units at {self.last_price}, stop at {stop} at time {buy_time}")
                    print("==================")

                    indicator_dict = {
                        'MACD': {'type': 'MACD', 'macd': self.daily_macd["MACD"], 'signal': self.daily_macd['SIGNAL'], 'histogram': self.daily_macd['HISTOGRAM']},
                        'Bullish divergence': {'type': 'below', 'data': self.divergences["regularBull"]},
                        'Bollinger Bands': {'type': 'bands',
                                      'lower': self.daily_bb.BB_LOWER,
                                      'upper': self.daily_bb.BB_UPPER,
                                      'mid': self.daily_bb.BB_MIDDLE,
                                      'mid_name': 'Bollinger Mid Line'
                        },
                        'MA_short': {'type': 'MA', 'data': TA.EMA(self.daily_data, 12)},
                        'MA_LONG': {'type': 'MA', 'data': TA.EMA(self.daily_data, 26)}
                    }

                    ap = AutoPlot(self.daily_data)
                    ap.plot(indicators=indicator_dict, instrument=f'{self.ticker}')

                    return Purchase(ticker=self.ticker, count=stock_count, isLong=True, stop=stop, buyPrice=self.last_price)
        if not is_dmi_up and self.day_impulse != "Green" and self.week_impulse != "Green" and self.daily_season == "Fall" and self.is_in_buy_value_town(False): # SHORT
            for i in range(len(self.divergences) - 1, len(self.divergences) - divergence_age, -1):
                if self.divergences["regularBear"][i] == 1.0:
                    stop = self.get_stop_value(price=self.last_price, is_long=False, atr_count=atr_count)
                    max_stock_count = math.floor(single_trade_max_loss / (self.last_price - stop))
                    stock_count = min(math.floor(cash/self.last_price), max_stock_count)

                    print("==================")
                    print(f"Hit Bearish Stock: {self.ticker}, short {stock_count} units at {self.last_price}, stop at {stop} at time {buy_time}")
                    print("==================")

                    indicator_dict = {
                        'MACD': {'type': 'MACD', 'macd': self.daily_macd["MACD"], 'signal': self.daily_macd['SIGNAL'], 'histogram': self.daily_macd['HISTOGRAM']},
                        'Bearish divergence': {'type': 'below', 'data': self.divergences['regularBear']},
                        'MA_short': {'type': 'MA', 'data': TA.EMA(self.daily_data, 12)},
                        'MA_LONG': {'type': 'MA', 'data': TA.EMA(self.daily_data, 26)}
                    }
                    ap = AutoPlot(self.daily_data)
                    ap.plot(indicators=indicator_dict, instrument=f'{self.ticker}')

                    return Purchase(ticker=self.ticker, count=stock_count, isLong=False, stop=stop, buyPrice=self.last_price)
        return None
    
    # TODO buy more than 1 eventually
    def add_to_winner(self, purchases: List[Purchase], cash: float) -> Optional[Purchase]:
        if cash < self.last_price or len(purchases) == 0:
            return None
        new_stop = 0
        new_purchase: Optional[Purchase] = None
        p = purchases[0]
        one_atr = self.get_stop_value(price=self.last_price, atr_count=1, is_long=p.isLong)
        new_stop = self.get_stop_value(price=self.last_price, atr_count=2, is_long=p.isLong)
        highest_long_buy = max([p.buyPrice if p.isLong else 0 for p in purchases])
        lowest_short_buy = min([p.buyPrice if not p.isLong else 0 for p in purchases])
        if p.isLong and self.last_price >= one_atr + highest_long_buy:
            new_purchase = Purchase(ticker=p.ticker, count=1, isLong=p.isLong, stop=new_stop, buyPrice=self.last_price)
        elif not p.isLong and self.last_price <= one_atr + lowest_short_buy:
            new_purchase = Purchase(ticker=p.ticker, count=1, isLong=p.isLong, stop=new_stop, buyPrice=self.last_price)
        
        if new_purchase is not None:
            print(f"Added to winner {self.ticker}, {new_purchase}")
            purchases.append(new_purchase)
            for p in purchases:
                p.stop = new_purchase.stop
        return new_purchase
    
    def calculate_season(self, df: pd.DataFrame):
        '''
        Calculates the season given a MACD
        '''
        is_positive=df['HISTOGRAM'].iloc[-1] - df['HISTOGRAM'].iloc[-2] > 0
        is_less_zero=df['HISTOGRAM'].iloc[-1] < 0

        if is_positive and is_less_zero:
            return "Spring"
        elif is_positive and not is_less_zero:
            return "Summer"
        elif not is_positive and is_less_zero:
            return "Winter"
        else:
            return "Autumn"
