from pydantic import BaseModel

class Purchase(BaseModel):
    ticker: str
    count: int
    isLong: bool
    stop: float
    buyPrice: float
