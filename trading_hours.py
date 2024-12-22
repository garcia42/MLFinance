from datetime import datetime, date, timedelta
import pytz
import pandas as pd

def is_nyse_trading_hours(row: pd.Series) -> bool:
    unix_timestamp = row.t
    # Convert the Unix timestamp to a datetime object in ET
    et_time = datetime.fromtimestamp(unix_timestamp/1000, pytz.timezone('US/Eastern'), )

    # Convert UTC time to Eastern Time (ET)
    # eastern = pytz.timezone('US/Eastern')
    # et_time = pytz.utc.localize(utc_time).astimezone(eastern)

    # Check if the day is a weekday (Monday=0, Sunday=6)
    if et_time.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False

    # Check if the time is within NYSE trading hours (9:30 AM to 4:00 PM ET)
    #TODO(Move this back to 9:30am and not 9am)
    market_open = et_time.replace(hour=9, minute=0, second=0, microsecond=0)
    market_close = et_time.replace(hour=16, minute=0, second=0, microsecond=0)

    if not (market_open <= et_time <= market_close):
        return False

    # Check for NYSE holidays (simple example, should include actual NYSE holiday dates)
    nyse_holidays = get_nyse_holidays(et_time.year)
    if et_time.date() in nyse_holidays:
        return False

    return True

def get_nyse_holidays(year) -> set:
    # This function should return a set of dates that are NYSE holidays for the given year.
    # Here, we'll add a few common holidays; in practice, you'd need a complete list.
    holidays = {
        # New Year's Day (if it falls on a weekend, it may be observed on a weekday)
        date(year, 1, 1),
        # Independence Day (4th of July, if on a weekend, may be observed)
        date(year, 7, 4),
        # Christmas Day (if on a weekend, may be observed)
        date(year, 12, 25),
        # ... add more holidays as needed
    }

    # Handle observed holidays if they fall on a weekend
    observed_holidays = set()
    for holiday in holidays:
        if holiday.weekday() == 5:  # Saturday
            observed_holidays.add(holiday + timedelta(days=-1))  # Observed on Friday
        elif holiday.weekday() == 6:  # Sunday
            observed_holidays.add(holiday + timedelta(days=1))  # Observed on Monday
    return holidays.union(observed_holidays)