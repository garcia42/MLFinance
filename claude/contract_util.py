# Standard library modules
from datetime import datetime

# Third-party modules
from ib_insync import Future


def get_next_quarterly_expiry():
    """Get the next quarterly expiration month."""
    current_month = datetime.now().month
    current_year = datetime.now().year
    
    # Quarterly months are 3,6,9,12
    quarterly_months = [3, 6, 9, 12]
    
    # Find next quarterly month
    for month in quarterly_months:
        if month > current_month:
            return f"{current_year}{month:02d}"
    
    # If we're past December, go to next year March
    return f"{current_year+1}03"

# # Create contract for next quarterly expiry
# contract = Future(
#     symbol='ES',
#     exchange='CME',
#     lastTradeDateOrContractMonth=get_next_quarterly_expiry()
# )

def get_local_symbol(date_str: str, symbol: str = "ES") -> str:
    """
    Convert YYYYMMDD to futures contract local symbol
    Example: '20240321' -> 'ESH4' for March 2024 ES contract
    
    Args:
        date_str (str): Date in YYYYMMDD format
        symbol (str): Future symbol (default: 'ES')
    
    Returns:
        str: Local symbol (e.g., 'ESH4')
    """
    # Month codes for futures contracts
    month_codes = {
        1: 'F',   # January
        2: 'G',   # February
        3: 'H',   # March
        4: 'J',   # April
        5: 'K',   # May
        6: 'M',   # June
        7: 'N',   # July
        8: 'Q',   # August
        9: 'U',   # September
        10: 'V',  # October
        11: 'X',  # November
        12: 'Z'   # December
    }
    
    # Extract year and month from date string
    year = int(date_str[2:4])  # Get last two digits of year
    month = int(date_str[4:6])
    
    # Get month code
    month_code = month_codes[month]
    
    # Combine components
    local_symbol = f"{symbol}{month_code}{year}"
    
    return local_symbol