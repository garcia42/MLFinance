import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import pyarrow as pa
import pyarrow.parquet as pq

class FeatureStorage:
    def __init__(self, storage_path: str = 'features.parquet'):
        self.storage_path = Path(storage_path)
        
    def load_existing_features(self) -> tuple[pd.DataFrame, datetime]:
        """Load existing features and return the latest date"""
        if not self.storage_path.exists():
            return pd.DataFrame(), datetime(2019, 1, 1)
        
        df = pd.read_parquet(self.storage_path)
        if df.empty:
            return df, datetime(2019, 1, 1)
        
        # Assuming index is datetime
        df.index = pd.to_datetime(df.index)
        latest_date = df.index.max()
        
        return df, latest_date
    
    def save_features(self, df: pd.DataFrame, partition_cols: list = None):
        """Save features to parquet with optional partitioning"""
        if partition_cols:
            table = pa.Table.from_pandas(df)
            pq.write_to_dataset(
                table,
                root_path=str(self.storage_path),
                partition_cols=partition_cols
            )
        else:
            df.to_parquet(self.storage_path)
            
    def update_features(self, new_data: pd.DataFrame):
        """Merge new data with existing data, removing duplicates"""
        existing_data, _ = self.load_existing_features()
        
        # Combine old and new data, keeping the newer version of duplicates
        combined_data = pd.concat([existing_data, new_data])
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
        combined_data = combined_data.sort_index()
        
        self.save_features(combined_data)
        return combined_data
    
    def get_missing_dates(self, start_date: datetime, end_date: datetime = None) -> tuple[datetime, datetime]:
        """Calculate what date range needs to be downloaded"""
        _, latest_stored_date = self.load_existing_features()
        
        if end_date is None:
            end_date = datetime.now()
            
        # If we have data after start_date, begin from the latest stored date, add some overlap
        if latest_stored_date > start_date:
            start_date = latest_stored_date - timedelta(days=1)
            
        return start_date, end_date
