# Standard library modules
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union, List

# Third-party modules
import pandas as pd
from ib_insync import *

# FinancialMachineLearning modules
from FinancialMachineLearning.barsampling.core import RunBarFeatures
from FinancialMachineLearning.filter.etf_trick import etfTrick

# Claude modules
from claude.contract_util import get_local_symbol


class ContinuousFuturesContract:
    def __init__(self, ib: IB):
        """
        Initialize the continuous future contract calculator
        
        Args:
            ib (IB): IB Gateway/TWS host connection
        """
        self.ib = ib
    
    def get_third_friday(self, date) -> pd.Timestamp:
        """Gets the pd.Timestamp of the third friday given a date of a month

        Args:
            date (_type_): Month to get the Timestamp for

        Returns:
            _type_: _description_
        """
        # Get the first day of the month
        first_day = pd.Timestamp(date).replace(day=1)
        
        # Find the first Friday (weekday=4)
        if first_day.weekday() <= 4:
            first_friday = first_day + pd.Timedelta(days=(4 - first_day.weekday()))
        else:
            first_friday = first_day + pd.Timedelta(days=(11 - first_day.weekday()))
        
        # Add 2 weeks to get to the third Friday
        third_friday = first_friday + pd.Timedelta(weeks=2)
        
        return third_friday
    
    def get_es_contract_files(self, lookback_days: int, base_date: datetime = None) -> list[str]:
        """
        Generate ES futures contract filenames for a specified lookback period.
        
        Args:
            lookback_days: Number of days to look back from base_date
            base_date: Reference date (defaults to today if None)
        
        Returns:
            List of filenames in format ES_[M]YY_1min.txt where M is the month code
        """
        # CME month codes
        MONTH_CODES = {
            3: 'H',  # March
            6: 'M',  # June
            9: 'U',  # September
            12: 'Z'  # December
        }
        
        if base_date is None:
            base_date = datetime.now()
        
        start_date = base_date - timedelta(days=lookback_days)
        
        # Create date range
        dates = pd.date_range(start=start_date, end=base_date, freq='D')
        
        contracts = set()  # Use set to avoid duplicates
        
        for date in dates:
            year = date.year
            month = date.month
            
            # Find the next valid contract month
            while month not in MONTH_CODES:
                month += 1
                if month > 12:
                    month = 3  # Roll to March of next year
                    year += 1
            
            # Format filename
            filename = f"ES_{MONTH_CODES[month]}{str(year)[-2:]}_1min.txt"
            contracts.add(filename)
        
        # Sort to ensure consistent order
        return sorted(list(contracts))
    
    def load_es_contract_data(
        self,
        contract_files: List[str],
        data_dir: Union[str, Path],
        start_date: datetime = None,
        end_date: datetime = None
    ) -> pd.DataFrame:
        """
        Load and combine ES futures contract data from multiple files.
        Creates a DataFrame with multiple contracts per date and identifies current contract.
        
        Args:
            contract_files: List of contract filenames
            data_dir: Directory containing the data files
            start_date: Optional start date filter
            end_date: Optional end date filter
        
        Returns:
            DataFrame with contract data including current contract identification
        """
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        data_dir = Path(data_dir)
        dfs = []
        
        columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dtypes = {
            'open': float,
            'high': float,
            'low': float,
            'close': float,
            'volume': float
        }
        
        # Load all contracts into separate DataFrames
        for file in contract_files:
            file_path = data_dir / file
            contract = file.split('_')[1]  # e.g., 'M08'
            logger.info(f"Reading contract {contract} from {file} at {file_path}")

            df = pd.read_csv(
                file_path,
                names=columns,
                dtype=dtypes,
                parse_dates=['date']
            )
            
            # Add contract identifier column
            df['sec_col'] = contract  # Changed from 'contract' to 'sec_col'
            
            # Apply date filters if provided
            if start_date is not None:
                df = df[df['date'] >= start_date]
            if end_date is not None:
                df = df[df['date'] <= end_date]
            
            dfs.append(df)
            logger.info(f"Successfully loaded {len(df)} rows from {file}")

        if not dfs:
            raise ValueError("No data was successfully loaded")
        
        # Combine all DataFrames
        combined_df = pd.concat(dfs, axis=0)
        
        # Sort by date and contract
        combined_df = combined_df.sort_values(['date', 'sec_col'])
        
        # Determine current contract for each date
        # This example uses the nearest contract as current
        def get_current_contract(group):
            # Logic to determine current contract
            # This is a simple example - you might want more sophisticated logic
            return group['sec_col'].iloc[0]
        
        current_contracts = combined_df.groupby('date').apply(get_current_contract)
        
        # Add current_sec_col column
        combined_df['current_sec_col'] = combined_df['date'].map(current_contracts)
        
        # Set index but keep date as column for easy access
        combined_df = combined_df.set_index('date', drop=False)
        combined_df = combined_df.rename(columns={'date': 'date_col'})
        
        # Remove any duplicates
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        
        logger.info(f"Final dataset contains {len(combined_df)} rows")
        
        return combined_df

    async def get_historical_contracts(self, underlying="ES", exchange="CME", lookback_days=365) -> list[Contract]:
        """
        Get both active and expired contracts within the lookback period
        
        Args:
            underlying (str): Future contract symbol
            exchange (str): Exchange identifier
            lookback_days (int): Number of days to look back
            
        Returns:
            list: List of contract objects
        """
        # Calculate the start date for our lookback period
        start_date = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=lookback_days)
        start_year = start_date.year
        current_year = pd.Timestamp.now(tz='UTC').year    
        contracts = []
        
        # Look for contracts from start_year to current_year + 1
        for year in range(start_year, current_year + 2):
            for month in range(1, 13):
                # For ES futures, typically only quarterly contracts (Mar, Jun, Sep, Dec)
                if underlying == "ES" and month not in [3, 6, 9, 12]:
                    continue

                # Create contract specification
                # last_trade_date = f"{year}{month}"
                last_trade_date = f"{year}{month:02d}"
                last_trade_date = self.get_third_friday(pd.Timestamp(last_trade_date + "20"))
                last_trade_date=last_trade_date.strftime("%Y%m%d")
                c = Contract(
                    secType="FUT",
                    includeExpired=True,
                    symbol=underlying,
                    exchange=exchange,
                    lastTradeDateOrContractMonth=last_trade_date,
                    currency="USD",
                    tradingClass=underlying,
                    multiplier=50
                )
                
                self.ib.qualifyContracts(c)
                
                contracts.append(c)

        return contracts

    async def get_historical_data(self, underlying="ES", exchange="SMART", lookback_days=365, end_date=pd.Timestamp.now(tz="UTC")) -> pd.DataFrame:
        all_data = []
        chunk_size = 60  # IB's typical limit for detailed data
        
        # Calculate date ranges with timezone awareness
        start_date = end_date - pd.Timedelta(days=lookback_days)

        # Get the list of active contracts from the true start to now
        contracts = await self.get_historical_contracts(underlying, exchange, lookback_days=((pd.Timestamp.now(tz="UTC") - start_date).days))
        contracts = self.get_es_contract_files(lookback_days=lookback_days)

        print(f"Fetching historic market data for {lookback_days} ago from {start_date} to {end_date}")
        
        # Generate list of date chunks
        date_chunks: list[list[pd.Timestamp]] = []
        current_end = end_date
        while current_end > start_date:
            chunk_end = current_end
            chunk_start = max(current_end - pd.Timedelta(days=chunk_size), start_date)
            date_chunks.append((chunk_start, chunk_end))
            current_end = chunk_start - pd.Timedelta(seconds=1)

        for contract in contracts:
            print(f"Collecting data for contract: {contract}")

            # try:
            # Parse and localize contract dates
            last_trade_date = contract.lastTradeDateOrContractMonth
            if len(contract.lastTradeDateOrContractMonth) == 5:
                last_trade_date = contract.lastTradeDateOrContractMonth[:-1] + "0" + contract.lastTradeDateOrContractMonth[-1]
            last_trade_date = self.get_third_friday(pd.Timestamp(last_trade_date + "20"))
            last_trade_date=last_trade_date.strftime("%Y%m%d")
            contract_expiry = pd.Timestamp(last_trade_date, tz="UTC")
            contract_start = contract_expiry - pd.Timedelta(days=365)  # Trading typically starts ~1 year before expiry

            # Skip if contract expires before our start date or starts after our end date
            if contract_expiry < start_date or contract_start > end_date:
                print(f"  Skipping contract {contract.localSymbol} (outside date range)")
                continue
            
            contract_data = []
            
            for chunk_start, chunk_end in date_chunks:
                # Skip chunks outside contract trading period
                if chunk_end < contract_start or chunk_start > contract_expiry:
                    continue
                
                ib_time_format = '%Y%m%d-%H:%M:%S'
                print(f"  Fetching chunk from {chunk_start.strftime(ib_time_format)} "
                    f"to {chunk_end.strftime(ib_time_format)} UTC")
                
                # Calculate the duration for this chunk
                duration_days = (chunk_end - chunk_start).days + 1
                
                # Format end time for IB request (in UTC)
                request_end_time = chunk_end.strftime(ib_time_format)
                
                bars = await self.ib.reqHistoricalDataAsync(
                    contract,
                    request_end_time,
                    f"{duration_days} D",
                    "5 mins",
                    "TRADES",
                    useRTH=True,
                    formatDate=1
                )
                
                if bars:
                    print(f"{len(bars)} Bars found for {contract.lastTradeDateOrContractMonth} from {chunk_start.strftime(ib_time_format)} to {chunk_end.strftime(ib_time_format)}")
                    for futures_bar in bars:
                        # Ensure bar date is timezone aware
                        if chunk_start <= futures_bar.date <= chunk_end:
                            contract_data.append({
                                'date': futures_bar.date,
                                'open': futures_bar.open,
                                'high': futures_bar.high,
                                'low': futures_bar.low,
                                'close': futures_bar.close,
                                'volume': futures_bar.volume,
                                'contract': get_local_symbol(contract.lastTradeDateOrContractMonth),
                                'current_contract': get_local_symbol(contract.lastTradeDateOrContractMonth)
                            })
                else:
                    print(f"No data found for {contract.lastTradeDateOrContractMonth} from {chunk_start.strftime(ib_time_format)} to {chunk_end.strftime(ib_time_format)}")
                await asyncio.sleep(1)  # Rate limiting between chunks

            # Deduplicate data for this contract
            if contract_data:
                df_contract = pd.DataFrame(contract_data)
                df_contract = df_contract.drop_duplicates(subset=['date'])
                all_data.extend(df_contract.to_dict('records'))

            await asyncio.sleep(2)  # Rate limiting between contracts

            # except Exception as e:
            #     print(f"Error processing contract {contract.localSymbol}: {e}")
            #     continue
        
        if not all_data:
            print("No data was collected for any contract")
            return pd.DataFrame()
        
        # Convert to DataFrame and clean up
        df = pd.DataFrame(all_data)
        
        # Sort by date and contract
        df = df.sort_values(['date', 'contract'])
        
        # Remove any duplicate entries
        df = df.drop_duplicates(subset=['date', 'contract'])
        
        # Verify data continuity
        for contract in df['contract'].unique():
            contract_data = df[df['contract'] == contract]
            if len(contract_data) > 0:
                date_diff = contract_data['date'].diff()
                # Look for gaps larger than 5 minutes during trading hours
                gaps = date_diff[date_diff > pd.Timedelta(minutes=5)]
                if not gaps.empty:
                    print(f"Warning: Found gaps in data for {contract}:")
                    for gap_start, gap_duration in zip(gaps.index, gaps):
                        gap_date = df.loc[gap_start, 'date']
                        print(f"  Gap of {gap_duration} at {gap_date}")
        
        return df
    
    async def get_continuous_contract(self, underlying="ES", exchange="CME", lookback_days=365, data_dir="Data/ES_1min") -> pd.DataFrame:
        """
        Generate continuous contract data with OHLCV data
        
        Args:
            underlying (str): Future contract symbol
            exchange (str): Exchange identifier
            lookback_days (int): Number of days to look back
            
        Returns:
            pd.DataFrame: Continuous contract data with OHLCV and roll adjustments
        """
        try:
            # Collect raw data
            start_date = datetime.now() - timedelta(days=lookback_days)

            # Define your desired date range
            contract_files = self.get_es_contract_files(365 * 5)

            # Load data
            raw_data = self.load_es_contract_data(
                contract_files=contract_files,
                data_dir=data_dir,
                start_date=start_date,
                end_date=datetime.now()
            )

            # Display sample of data
            print("\nSample of loaded data:")
            print(raw_data.head())
            
            etf_trick = etfTrick.get_futures_roll_series(
                data_df=raw_data,
                open_col='open',
                close_col='close',
                sec_col='sec_col',
                current_sec_col='current_sec_col',
                roll_backward=True
            )

            # Combine all series into a single DataFrame with explicit date index
            raw_data['close'] = raw_data['close'] + etf_trick

            # Set and sort the date index
            result = raw_data.sort_index()
            run_bar = RunBarFeatures(result)
            run_bar_df = run_bar.ema_dollar_run_bar()[0]
            print("\nSample of run bars:")
            print(run_bar_df.head())
            run_bar_df = run_bar_df.set_index(pd.to_datetime(run_bar_df['date_time']))
            return run_bar_df

        except Exception as e:
            print(f"Error generating continuous contract: {e}")
            raise