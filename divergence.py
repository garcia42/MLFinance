import numpy as np
import pandas as pd
from typing import Union

def unroll_signal_list(signals: Union[list, pd.Series]) -> np.array:
    """Unrolls a rolled signal list.

    Parameters
    ----------
    signals : Union[list, pd.Series]
        DESCRIPTION.

    Returns
    -------
    unrolled_signals : np.array
        The unrolled signal series.

    See Also
    --------
    This function is the inverse of rolling_signal_list.

    Examples
    --------
    >>> unroll_signal_list([0, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1])
        array([ 0.,  1.,  0.,  0.,  0., -1.,  0.,  0.,  1.,  0.,  0.])

    """
    unrolled_signals = np.zeros(len(signals))
    for i in range(1, len(signals)):
        if signals[i] != signals[i - 1]:
            unrolled_signals[i] = signals[i]

    if isinstance(signals, pd.Series):
        unrolled_signals = pd.Series(data=unrolled_signals, index=signals.index)

    return unrolled_signals

def candles_between_crosses(
    crosses: Union[list, pd.Series], initial_count: int = 0
) -> Union[list, pd.Series]:
    """Returns a rolling sum of candles since the last cross/signal occurred.

    Parameters
    ----------
    crosses : list | pd.Series
        The list or Series containing crossover signals.

    Returns
    -------
    counts : TYPE
        The rolling count of bars since the last crossover signal.

    See Also
    ---------
    indicators.crossover
    """

    count = 0
    counts = []

    for i in range(len(crosses)):
        if crosses[i] == 0:
            # Change in signal - reset count
            count += 1
        else:
            count = initial_count

        counts.append(count)

    if isinstance(crosses, pd.Series):
        # Convert to Series
        counts = pd.Series(data=counts, index=crosses.index, name="counts")

    return counts

def merge_signals(signal_1: list, signal_2: list) -> list:
    """Returns a single signal list which has merged two signal lists.

    Parameters
    ----------
    signal_1 : list
        The first signal list.

    signal_2 : list
        The second signal list.

    Returns
    -------
    merged_signal_list : list
        The merged result of the two inputted signal series.

    Examples
    --------
    >>> s1 = [1,0,0,0,1,0]
    >>> s2 = [0,0,-1,0,0,-1]
    >>> merge_signals(s1, s2)
        [1, 0, -1, 0, 1, -1]

    """
    merged_signal_list = signal_1.copy()
    for i in range(len(signal_1)):
        if signal_2[i] != 0:
            merged_signal_list[i] = signal_2[i]

    return merged_signal_list


def rolling_signal_list(signals: Union[list, pd.Series]) -> list:
    """Returns a list which repeats the previous signal, until a new
    signal is given.

    Parameters
    ----------
    signals : list | pd.Series
        A series of signals. Zero values are treated as 'no signal'.

    Returns
    -------
    list
        A list of rolled signals.

    Examples
    --------
    >>> rolling_signal_list([0,1,0,0,0,-1,0,0,1,0,0])
        [0, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1]

    """
    rolling_signals = [0]
    last_signal = rolling_signals[0]

    if isinstance(signals, list):
        for i in range(1, len(signals)):
            if signals[i] != 0:
                last_signal = signals[i]
            rolling_signals.append(last_signal)
    else:
        for i in range(1, len(signals)):
            if signals.iloc[i] != 0:
                last_signal = signals.iloc[i]
            rolling_signals.append(last_signal)

    if isinstance(signals, pd.Series):
        rolling_signals = pd.Series(data=rolling_signals, index=signals.index)

    return rolling_signals

def ema(data: pd.DataFrame, period: int = 14, smoothing: int = 2) -> list:
    """Exponential Moving Average."""
    ema = [sum(data[:period]) / period]
    for price in data[period:]:
        ema.append(
            (price * (smoothing / (1 + period)))
            + ema[-1] * (1 - (smoothing / (1 + period)))
        )
    for i in range(period - 1):
        ema.insert(0, np.nan)
    return ema

def find_swings(data: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """Locates swings in the inputted data using a moving average gradient
    method.

    Parameters
    ----------
    data : pd.DataFrame | pd.Series | list | np.array
        An OHLC dataframe of price, or an array/list/Series of data from an
        indicator (eg. RSI).

    n : int, optional
        The moving average period. The default is 2.

    Returns
    -------
    swing_df : pd.DataFrame
        A dataframe containing the swing levels detected.

    pd.Series(hl2, name="hl2"),
    """
    # Prepare data
    if isinstance(data, pd.DataFrame):
        # OHLC data
        hl2 = (data["High"].values + data["Low"].values) / 2
        swing_data = pd.Series(ema(hl2, n), index=data.index)
        low_data = data["Low"].values
        high_data = data["High"].values

    elif isinstance(data, pd.Series):
        # Pandas series data
        swing_data = pd.Series(ema(data.fillna(0), n), index=data.index)
        low_data = data
        high_data = data

    else:
        # Find swings in alternative data source
        data = pd.Series(data)

        # Define swing data
        swing_data = pd.Series(ema(data, n), index=data.index)
        low_data = data
        high_data = data

    signed_grad = np.sign((swing_data - swing_data.shift(1)).bfill())
    swings = (signed_grad != signed_grad.shift(1).bfill()) * -signed_grad

    # Calculate swing extrema
    lows = []
    highs = []
    for i, swing in enumerate(swings):
        if swing < 0:
            # Down swing, find low price
            highs.append(0)
            lows.append(min(low_data[i - n + 1 : i + 1]))
        elif swing > 0:
            # Up swing, find high price
            highs.append(max(high_data[i - n + 1 : i + 1]))
            lows.append(0)
        else:
            # Price movement
            highs.append(0)
            lows.append(0)

    # Determine last swing
    trend = rolling_signal_list(-swings)
    swings_list = merge_signals(lows, highs)
    last_swing = rolling_signal_list(swings_list)

    # Need to return both a last swing low and last swing high list
    last_low = rolling_signal_list(lows)
    last_high = rolling_signal_list(highs)

    swing_df = pd.DataFrame(
        data={"Highs": last_high, "Lows": last_low, "Last": last_swing, "Trend": trend},
        index=swing_data.index,
    )

    return swing_df


def classify_swings(swing_df: pd.DataFrame, tol: int = 0) -> pd.DataFrame:
    """Classifies a dataframe of swings (from find_swings) into higher-highs,
    lower-highs, higher-lows and lower-lows.


    Parameters
    ----------
    swing_df : pd.DataFrame
        The dataframe returned by find_swings.

    tol : int, optional
        The classification tolerance. The default is 0.

    Returns
    -------
    swing_df : pd.DataFrame
        A dataframe containing the classified swings.
    """
    # Create copy of swing dataframe
    swing_df = swing_df.copy()

    new_level = np.where(swing_df.Last != swing_df.Last.shift(), 1, 0)

    candles_since_last = candles_between_crosses(new_level, initial_count=1)

    # Add column 'candles since last swing' CSLS
    swing_df["CSLS"] = candles_since_last

    # Find strong Support and Resistance zones
    swing_df["Support"] = (swing_df.CSLS > tol) & (swing_df.Trend == 1)
    swing_df["Resistance"] = (swing_df.CSLS > tol) & (swing_df.Trend == -1)

    # Find higher highs and lower lows
    swing_df["Strong_lows"] = (
        swing_df["Support"] * swing_df["Lows"]
    )  # Returns high values when there is a strong support
    swing_df["Strong_highs"] = (
        swing_df["Resistance"] * swing_df["Highs"]
    )  # Returns high values when there is a strong support

    # Remove duplicates to preserve indexes of new levels
    swing_df["FSL"] = unroll_signal_list(
        swing_df["Strong_lows"]
    )  # First of new strong lows
    swing_df["FSH"] = unroll_signal_list(
        swing_df["Strong_highs"]
    )  # First of new strong highs

    # Now compare each non-zero value to the previous non-zero value.
    low_change = np.sign(swing_df.FSL) * (
        swing_df.FSL
        - swing_df.Strong_lows.replace(0, np.nan).ffill().shift()
    )
    high_change = np.sign(swing_df.FSH) * (
        swing_df.FSH
        - swing_df.Strong_highs.replace(0, np.nan).ffill().shift()
    )

    # the first low_change > 0.0 is not a HL
    r_hl = []
    first_valid_idx = -1
    for i in low_change.index:
        v = low_change[i]
        if first_valid_idx == -1 and not np.isnan(v) and v != 0.0:
            first_valid_idx = i
        if first_valid_idx != -1 and i > first_valid_idx and v > 0.0:
            hl = True
        else:
            hl = False
        r_hl.append(hl)

    # the first high_change < 0.0 is not a LH
    r_lh = []
    first_valid_idx = -1
    for i in high_change.index:
        v = high_change[i]
        if first_valid_idx == -1 and not np.isnan(v) and v != 0.0:
            first_valid_idx = i
        if first_valid_idx != -1 and i > first_valid_idx and v < 0.0:
            lh = True
        else:
            lh = False
        r_lh.append(lh)

    swing_df["LL"] = np.where(low_change < 0, True, False)
    # swing_df["HL"] = np.where(low_change > 0, True, False)
    swing_df["HL"] = r_hl
    swing_df["HH"] = np.where(high_change > 0, True, False)
    # swing_df["LH"] = np.where(high_change < 0, True, False)
    swing_df["LH"] = r_lh

    return swing_df


def detect_divergence(
    classified_price_swings: pd.DataFrame,
    classified_indicator_swings: pd.DataFrame,
    tol: int = 2,
    method: int = 0,
) -> pd.DataFrame:
    """Detects divergence between price swings and swings in an indicator.

    Parameters
    ----------
    classified_price_swings : pd.DataFrame
        The output from classify_swings using OHLC data.

    classified_indicator_swings : pd.DataFrame
        The output from classify_swings using indicator data.

    tol : int, optional
        The number of candles which conditions must be met within. The
        default is 2.

    method : int, optional
        The method to use when detecting divergence (0 or 1). The default is 0.

    Raises
    ------
    Exception
        When an unrecognised method of divergence detection is requested.

    Returns
    -------
    divergence : pd.DataFrame
        A dataframe containing divergence signals.

    Notes
    -----
    Options for the method include:
        0: use both price and indicator swings to detect divergence (default)

        1: use only indicator swings to detect divergence (more responsive)
    """
    regular_bullish = []
    regular_bearish = []
    hidden_bullish = []
    hidden_bearish = []

    if method == 0:
        for i in range(len(classified_price_swings)):
            # Look backwards in each

            # REGULAR BULLISH DIVERGENCE
            if (
                sum(classified_price_swings["LL"][i - tol + 1 : i + 1])
                + sum(classified_indicator_swings["HL"][i - tol + 1 : i + 1])
                > 1
            ):
                regular_bullish.append(True)
            else:
                regular_bullish.append(False)

            # REGULAR BEARISH DIVERGENCE
            if (
                sum(classified_price_swings["HH"][i - tol + 1 : i + 1])
                + sum(classified_indicator_swings["LH"][i - tol + 1 : i + 1])
                > 1
            ):
                regular_bearish.append(True)
            else:
                regular_bearish.append(False)

            # HIDDEN BULLISH DIVERGENCE
            if (
                sum(classified_price_swings["HL"][i - tol + 1 : i + 1])
                + sum(classified_indicator_swings["LL"][i - tol + 1 : i + 1])
                > 1
            ):
                hidden_bullish.append(True)
            else:
                hidden_bullish.append(False)

            # HIDDEN BEARISH DIVERGENCE
            if (
                sum(classified_price_swings["LH"][i - tol + 1 : i + 1])
                + sum(classified_indicator_swings["HH"][i - tol + 1 : i + 1])
                > 1
            ):
                hidden_bearish.append(True)
            else:
                hidden_bearish.append(False)

        divergence = pd.DataFrame(
            data={
                "regularBull": unroll_signal_list(regular_bullish),
                "regularBear": unroll_signal_list(regular_bearish),
                "hiddenBull": unroll_signal_list(hidden_bullish),
                "hiddenBear": unroll_signal_list(hidden_bearish),
            },
            index=classified_price_swings.index,
        )
    elif method == 1:
        # Use indicator swings only to detect divergence
        # for i in range(len(classified_price_swings)):
        if True:
            price_at_indi_lows = (
                classified_indicator_swings["FSL"] != 0
            ) * classified_price_swings["Lows"]
            price_at_indi_highs = (
                classified_indicator_swings["FSH"] != 0
            ) * classified_price_swings["Highs"]

            # Determine change in price between indicator lows
            price_at_indi_lows_change = np.sign(price_at_indi_lows) * (
                price_at_indi_lows
                - price_at_indi_lows.replace(0, np.nan).ffill().shift()
            )
            price_at_indi_highs_change = np.sign(price_at_indi_highs) * (
                price_at_indi_highs
                - price_at_indi_highs.replace(0, np.nan).ffill().shift()
            )

            # DETECT DIVERGENCES
            regular_bullish = (classified_indicator_swings["HL"]) & (
                price_at_indi_lows_change < 0
            )
            regular_bearish = (classified_indicator_swings["LH"]) & (
                price_at_indi_highs_change > 0
            )
            hidden_bullish = (classified_indicator_swings["LL"]) & (
                price_at_indi_lows_change > 0
            )
            hidden_bearish = (classified_indicator_swings["HH"]) & (
                price_at_indi_highs_change < 0
            )

        divergence = pd.DataFrame(
            data={
                "regularBull": regular_bullish,
                "regularBear": regular_bearish,
                "hiddenBull": hidden_bullish,
                "hiddenBear": hidden_bearish,
            },
            index=classified_price_swings.index,
        )

    else:
        raise Exception("Error: unrecognised method of divergence detection.")

    return divergence


def autodetect_divergence(
    ohlc: pd.DataFrame,
    indicator_data: pd.DataFrame,
    tolerance: int = 1,
    method: int = 0,
) -> pd.DataFrame:
    """A wrapper method to automatically detect divergence from inputted OHLC price
    data and indicator data.

    Parameters
    ----------
    ohlc : pd.DataFrame
        A dataframe of OHLC price data.

    indicator_data : pd.DataFrame
        dataframe of indicator data.

    tolerance : int, optional
        A parameter to control the lookback when detecting divergence.
        The default is 1.

    method : int, optional
        The divergence detection method. Set to 0 to use both price and
        indicator swings to detect divergence. Set to 1 to use only indicator
        swings to detect divergence. The default is 0.

    Returns
    -------
    divergence : pd.DataFrame
        A DataFrame containing columns 'regularBull', 'regularBear',
        'hiddenBull' and 'hiddenBear'.

    See Also
    --------
    autotrader.indicators.find_swings
    autotrader.indicators.classify_swings
    autotrader.indicators.detect_divergence

    """

    # Price swings
    price_swings = find_swings(ohlc)
    price_swings_classified = classify_swings(price_swings)

    # Indicator swings
    indicator_swings = find_swings(indicator_data)
    indicator_classified = classify_swings(indicator_swings)

    # Detect divergence
    divergence = detect_divergence(
        price_swings_classified, indicator_classified, tol=tolerance, method=method
    )

    return divergence
