# Standard library modules
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

# Third-party modules
import fredapi
import nest_asyncio
import numpy as np
import pandas as pd
from ib_insync import *

# FinancialMachineLearning modules
from FinancialMachineLearning.filter.filter import cusum_filter
from FinancialMachineLearning.features.fracdiff import FractionalDifferentiatedFeatures

# Claude modules
from claude.contract_util import get_next_quarterly_expiry
from claude.feature_storage import FeatureStorage
from claude.fred_collector import FredDataCollector, create_fred_collector
from claude.train_model import label_and_analyze
from claude.validate import label_and_cross_validate, calculate_psr, build_model
from claude.continuous_futures_contract import ContinuousFuturesContract
from claude.market_indicator_data import fetch_market_indicators

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_es_contract(ib: IB):
    """Create an ES futures contract object"""
    contract = Future('ES', exchange='CME', lastTradeDateOrContractMonth=get_next_quarterly_expiry())
    
    all_contracts = ib.qualifyContracts(contract)

    # Get front-month contract
    print(all_contracts[0])
    return all_contracts[0] if all_contracts else None

def collect_all_historical_data(lookback_days: int, data_dir="Data/ES_1min") -> pd.DataFrame:
    """
    Collect and integrate all historical data
    
    Returns:
        DataFrame with market, macro, and sentiment data aligned
    """
    try:
        # Initialize FRED
        fred = fredapi.Fred(api_key='c44011f35ea9b58dc265ab237efaa525')

        # Initialize IB
        ib = IB()
        try:
            # Connect to IB
            ib.connect(host='127.0.0.1', port=7497, clientId=1)
        except Exception as e:
            logger.error(f"Error in data collection: {e}")

        # Collect market data
        continuous = ContinuousFuturesContract(ib)
        _, market_data = asyncio.get_event_loop().run_until_complete(continuous.get_continuous_contract(underlying="ES", exchange="CME", lookback_days=lookback_days, data_dir=data_dir))

        # Align all data to market data timestamps
        aligned_data = market_data.copy()

        # fc = create_fred_collector(
        #     fred_api=fred,
        #     lookback_days=lookback_days
        # )

        # # Collect macro data
        # macro_data: pd.DataFrame = fc.get_historical_fred_data()
        # if not macro_data.empty:
        #     macro_features = macro_data.reindex(aligned_data.index, method='ffill')
        #     aligned_data = pd.concat([aligned_data, macro_features], axis=1)

        # Fetch the market data (prices)
        # market_indicators_data = fetch_market_indicators(datetime.now() -  timedelta(days=lookback_days))
        # market_indicators_data = market_indicators_data.reindex(aligned_data.index, method='ffill')
        # aligned_data = pd.concat([aligned_data, market_indicators_data], axis=1)
        
        # Reset the index
        aligned_data.reset_index(inplace=True, drop=True)
        return aligned_data

    except Exception as e:
        logger.error(f"Error collecting integrated historical data: {e}")
        raise

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

def create_global_market_factor(data, indices=['^AXJO', '^GDAXI', '^FTSE']):
    """
    Creates a global market factor using PCA on major market indices.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing market indices
    indices : list
        List of column names for indices to include
        
    Returns:
    --------
    pd.Series
        PCA-transformed global market factor with same index as input data
    """
    # Extract just the indices we want
    market_data = data[indices].copy()
    
    # Handle any missing values
    market_data = market_data.fillna(method='ffill').fillna(method='bfill')
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(market_data)
    
    # Apply PCA
    pca = PCA(n_components=1)
    global_factor = pca.fit_transform(scaled_data)
    
    # Convert to Series with datetime index
    global_factor_series = pd.Series(
        global_factor.flatten(),
        index=data.index,
        name='global_market_factor'
    )
    
    # Store explained variance for reference
    explained_variance = pca.explained_variance_ratio_[0]
    print(f"Global market factor explains {explained_variance:.2%} of variance")
    
    # Store component weights for reference
    component_weights = pd.Series(
        pca.components_[0],
        index=indices
    )
    print("\nComponent weights:")
    print(component_weights)
    
    return global_factor_series

def cusum_filtering(features: pd.DataFrame):
    cusum_events = cusum_filter(
        features.close,
        threshold = 1.5,
        time_stamps = False
    )
    cusum_filtered = features.loc[cusum_events]
    return cusum_filtered

def run_data_collection(lookback_days = 15 * 365):

    storage = FeatureStorage('./Data/financial_features.parquet')
    historic_data = storage.load_existing_features()[0]
    
    historic_data = pd.read_parquet('./Data/financial_features.parquet')

    # Collect data
    historic_data = collect_all_historical_data(lookback_days)
    
    # historic_data = add_all_indicators(df=historic_data)
    
    historic_data = cusum_filtering(historic_data)

    historic_data = FractionalDifferentiatedFeatures.frac_diff_parallel(features=historic_data)

    # Update storage with new features
    return storage.update_features(historic_data)

if __name__ == '__main__':
    features = run_data_collection(lookback_days = 15 * 365)
    model = build_model(features=features, use_cache=False)
    # psr = calculate_psr(model)
    # print("PSR:", psr)
    # label_and_analyze(features=None)
    # label_and_analyze(features=features)
    label_and_cross_validate(features=features)
    