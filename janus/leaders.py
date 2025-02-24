from indicators.technical_indicators import multi_indicators
from indicators.Data.market_indicator_data import fetch_market_indicators

import asyncio

from ib_insync import IB

async def get_positive_leaders(ib: IB):
    market_indicators_data, n_markets = await fetch_market_indicators(ib=ib, lookback_years=2)
    # market_indicators_data, n_markets = await fetch_market_indicators(ib=ib, lookback_years=2)
    _, janus, _, market_names = multi_indicators(df=market_indicators_data, n_markets=n_markets, lookback=100)
    
    leader_indices, leader_doms, cma = janus.get_current_leader_dom_vs_cma()

    leaders_to_buy = []
    # Compare each leader's DOM to CMA
    for idx, dom in zip(leader_indices, leader_doms):
        print(f"Market {market_names[idx]}: DOM = {dom}, CMA = {cma}, Above CMA? {dom > cma}")
        if dom > cma:
            leaders_to_buy.append(market_names[idx])

    return leaders_to_buy