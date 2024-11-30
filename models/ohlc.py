from typing import List
from pydantic import BaseModel

class OHLC(BaseModel):
    ticker: str
    data: List[dict]