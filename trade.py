'''
Functions for trading
'''

from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple
import math

import pandas as pd
from finta import TA
from urllib3 import HTTPResponse
from polygon import RESTClient
from convert_ohlc import convert_data_daily, convert_data_weekly

from models.securitydata import Purchase, SecurityData
from models.ohlc import OHLC
import divergence
from trading_hours import is_nyse_trading_hours

def get_stock_data(polygon_client: RESTClient, ticker: str, time_span: str, start_date, end_date) -> Optional[pd.DataFrame]:
    '''
    Return stock data for a stock
    '''
    hour_data = polygon_client.get_aggs(ticker=ticker, multiplier=1, from_=start_date, to=end_date, sort="asc", limit=50000, timespan=time_span) # Max 83 hours
    if isinstance(hour_data, HTTPResponse):
        print(f"Failed to retrieve data {hour_data}")
        return None
    hour_data_aggs = [
        {
            "Open": agg.open,
            "High": agg.high,
            "Low": agg.low,
            "Close": agg.close,
            "Volume": float(agg.volume),
            "vwap": agg.vwap,
            "t": agg.timestamp,
            # "transactions": agg.transactions,
        }
        for agg in hour_data
    ]
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(hour_data_aggs)
    if len(df) == 0:
        print(f"Empty DF for ticker {ticker} not enough data")
        return None
    df["date"] = pd.to_datetime(df['t'], unit='ms')
    df.set_index("date",drop=False, inplace=True)
    df.index = df.index.tz_localize(timezone.utc)
    df.sort_values(by="t")
    return df

def get_stock_tickers(polygon_client: RESTClient, limit: int, letter: str) -> List[str]:
    '''
    Get stock tickers from NASDAQ between letters.
    '''
    tickers = polygon_client.list_tickers(market="stocks", type="CS", limit=limit, ticker_gte=letter[0], ticker_lte=letter[1])
    if isinstance(tickers, HTTPResponse):
        print(f"Error downloading tickers {tickers}")
        return None

    ticker_list = []
    i = 0
    while True and i < limit:
        i += 1
        try:
            cur = next(tickers)
        except StopIteration:
            break
        else:
            ticker_list.append(cur.ticker)

    return ticker_list

# def download_stock_data(polygon_client: RESTClient, t: str) -> Optional[OHLC]:
def download_stock_data(polygon_client: RESTClient, t: str) -> Optional[pd.DataFrame]:
    print(f"Downloading {t}")
    today = datetime.today() + timedelta(days=1)
    df = pd.DataFrame()
    time_jump_days = 3
    while today > datetime.today() - timedelta(days=710):
        days_ago_dt = (today - timedelta(days=time_jump_days)).strftime('%Y-%m-%d')
        new_df = get_stock_data(polygon_client, t, "hour", days_ago_dt, today.strftime('%Y-%m-%d'))
        if new_df is None:
            return None
        df = pd.concat([df, new_df])
        today = today - timedelta(days=time_jump_days)
    if df is not None and len(df) > 3360: #1680 trading hours in a year
        df = df.drop_duplicates().sort_values(by="t")
        print("Len(df):", len(df))
        # Keep only rows that are in the NYSE hours, this still keeps half of 9am and 1pm but that's another issue
        mask = df.apply(is_nyse_trading_hours, axis=1)
        df = df[~mask]
        print("Len(df) after removing out of hours:", len(df))
        return df
        # return OHLC(ticker=t, data=df.to_dict(orient="records"))
    else:
        print(f"Not enough data for {t}: {len(df)}")
        print(df.head(2))
        print(df.tail(2))
    return None

def create_sec_data(data: OHLC) -> Optional[SecurityData]:
    hourly_data = pd.DataFrame(data.data)
    daily_data = convert_data_daily(hourly_data)
    weekly_data = convert_data_weekly(hourly_data)

    week_ema = TA.EMA(weekly_data, 9)

    daily_macd = TA.MACD(daily_data)
    daily_macd["HISTOGRAM"] = daily_macd["MACD"] - daily_macd["SIGNAL"]
    daily_macd["date"] = daily_data["date"]
    daily_macd.set_index("date",drop=False, inplace=True)

    divergences = divergence.autodetect_divergence(daily_data, daily_macd["HISTOGRAM"])
    return SecurityData(ticker=data.ticker,
                        hourly_data=hourly_data,
                        weekly_ema=week_ema,
                        divergences=divergences,
                        daily_macd=daily_macd,
                        daily_data = daily_data)

def check_for_stop_hits(cash: int, purchases: List[Purchase], sec_data: SecurityData) -> Tuple[Dict[str, List[Purchase]], int, int]:
    '''
    '''
    loss = 0
    stops_hit = 0
    purchases_copy: List[Purchase] = []
    for p in purchases or []:
        df = sec_data.hourly_data
        # Determine the condition and values based on `isLong`
        if p.isLong:
            stop_loss_hit = df['Low'].iloc[-1] <= p.stop
            stop_price = df['Low'].iloc[-1]
            cash_change = p.stop * p.count
            loss_change = (p.buyPrice - p.stop) * p.count
        else:
            stop_loss_hit = df['High'].iloc[-1] >= p.stop
            stop_price = df['High'].iloc[-1]
            cash_change = -p.stop * p.count
            loss_change = (p.stop - p.buyPrice) * p.count

        if stop_loss_hit:
            stops_hit += 1
            long = "Long" if p.isLong else "Short"
            cur_time = sec_data.hourly_data["t"].iloc[-1]
            print(f"{long} Stop Loss Hit {p.ticker}: cash {cash} stop {p.stop} count {p.count}", "Stop price:", stop_price, "time:", cur_time)
            cash += cash_change
            loss += loss_change
        else:
            purchases_copy.append(p)

    return purchases_copy, cash, loss, stops_hit

def adjust_stops(purchases: List[Purchase], atr_count: float, sec_data: SecurityData):
    '''
    Look through all purchases and adjust stop to new ATR
    '''
    if not purchases:
        return

    p = purchases[0]
    new_stop = sec_data.get_stop_value(price=sec_data.last_price, atr_count=atr_count, is_long=p.isLong)

    # Determine if adjustment is needed
    adjustment_needed = (p.isLong and p.stop < new_stop) or (not p.isLong and p.stop > new_stop)

    if adjustment_needed:
        print(f"Stop adjusted for {p.ticker} from {p.stop} to {new_stop}, lastPrice: {sec_data.last_price}")
        
        # Apply the new stop value to all purchases
        for p in purchases:
            p.stop = new_stop

def single_trade_max_dollar_risk(max_risk: float,
                                 single_max_risk: float,
                                 all_sec_data: Dict[str, OHLC],
                                 cash: int,
                                 purchases: Dict[str, List[Purchase]],
                                 loss_this_month: float) -> float:
    '''
    Max risk to have for a single stock.
    '''
    total_risked = 0
    total_stock_value = 0
    for t, p_list in purchases.items():
        for p in p_list:
            total_stock_value += p.count * pd.DataFrame(all_sec_data.get(t).data)["Close"].iloc[-1]
            if p.isLong:
                # Buy at 100, stop at 90 = 100 - 90 = 10 at risk
                # Buy at 100, stop at 110 = 100 - 110 = -10
                total_risked += max(p.buyPrice - p.stop, 0)
            else:
                # Buy at 100, stop at 110 = 110 - 100 = 10 risk
                # Buy at 100, stop at 90 = 90 - 100 = -10 risk
                total_risked += max(p.stop - p.buyPrice, 0)

    if (total_risked + loss_this_month) > (total_stock_value + cash) * max_risk:
        return 0
    
    available_cash = total_short_positions(purchases=purchases) + cash

    # print(f"single_trade_max_dollar_risk min({single_max_risk} * ({total_stock_value} + {cash}), {available_cash})")
    return min(single_max_risk * (total_stock_value + cash), available_cash)

def total_short_positions(purchases: Dict[str, List[Purchase]]) -> int:
    total = 0
    for t, p_list in purchases.items():
        total += sum(p.count * p.buyPrice for p in p_list)
    return total

def total_account_equity(purchases: Dict[str, List[Purchase]],
                         cash: float,
                         all_sec_data: Dict[str, OHLC]):
    '''
    Account against all cash and purchases.
    '''
    total_assets = 0
    for t, p_list in purchases.items():
        total_assets += sum(p.count * pd.DataFrame(all_sec_data[t].data)["Close"].iloc[-1] for p in p_list)
    return total_assets + cash
