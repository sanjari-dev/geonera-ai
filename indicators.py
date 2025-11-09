# file: ai/indicators.py

import pandas as pd
import numpy as np
import math
import logging
from numba import njit, prange
from typing import Iterable


@njit
def _mad_numba(window: np.ndarray) -> float:
    """Mean absolute deviation around the window mean (for CCI)."""
    n = window.size
    if n == 0:
        return np.nan
    m = 0.0
    for i in range(n):
        m += window[i]
    m /= n
    s = 0.0
    for i in range(n):
        v = window[i] - m
        if v < 0:
            v = -v
        s += v
    return s / n


@njit
def _get_median(window: np.ndarray) -> float:
    """Numba helper for np.median (for Median Filter)."""
    return float(np.median(window))


@njit
def _periods_since_extrema_deque(arr: np.ndarray, period: int, is_max: bool) -> np.ndarray:
    """
    Calculates 'age' (i - idx_extrema) at each i for a sliding window of size period.
    is_max=True for argmax; False for argmin.
    """
    n = arr.size
    out = np.full(n, np.nan, dtype=np.float32)
    if period <= 1 or n == 0:
        return out

    dq = np.empty(n, dtype=np.int64)
    head = 0
    tail = 0

    def dq_clear():
        nonlocal head, tail
        head = 0
        tail = 0

    def dq_push(current_index):
        nonlocal head, tail
        while tail > head:
            j = dq[tail - 1]
            if is_max:
                if arr[j] >= arr[current_index]:
                    break
            else:
                if arr[j] <= arr[current_index]:
                    break
            tail -= 1
        dq[tail] = current_index
        tail += 1

    def dq_pop_front_older_than(left_bound):
        nonlocal head, tail
        while tail > head and dq[head] < left_bound:
            head += 1

    dq_clear()
    for i in range(n):
        dq_push(i)
        left = i - period + 1
        if left < 0:
            continue
        dq_pop_front_older_than(left)
        if tail > head:
            idx = dq[head]
            out[i] = np.float32(i - idx)
    return out


@njit(parallel=True)
def _compute_aroon_all(highs: np.ndarray, lows: np.ndarray, periods: np.ndarray):
    """
    Calculates Aroon Up/Down for all periods in parallel.
    """
    n = highs.size
    k = periods.size
    up = np.full((n, k), np.nan, dtype=np.float32)
    down = np.full((n, k), np.nan, dtype=np.float32)
    for pi in prange(k):
        p = int(periods[pi])
        if p <= 1:
            continue
        div = float(p - 1)
        if div == 0:
            continue
        ps_high = _periods_since_extrema_deque(highs, p, True)  # argmax age
        ps_low = _periods_since_extrema_deque(lows, p, False)  # argmin age
        for i in range(n):
            if np.isnan(ps_high[i]) or np.isnan(ps_low[i]):
                continue
            up_val = ((div - ps_high[i]) / div) * 100.0
            down_val = ((div - ps_low[i]) / div) * 100.0
            up[i, pi] = up_val
            down[i, pi] = down_val
    return up, down


def _get_typical_price(df: pd.DataFrame) -> pd.Series:
    return (df["high"].astype("float32") + df["low"].astype("float32") + df["close"].astype("float32")) / 3.0


def _get_total_volume(df: pd.DataFrame) -> pd.Series:
    return df['total_bid_volume'] + df['total_ask_volume']


def _get_true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df['close'].shift(1)
    high_low = df['high'] - df['low']
    high_prev_close = (df['high'] - prev_close).abs()
    low_prev_close = (df['low'] - prev_close).abs()
    tr_df = pd.DataFrame({
        'high_low': high_low,
        'high_prev_close': high_prev_close,
        'low_prev_close': low_prev_close
    })
    return tr_df.max(axis=1)


def _get_money_flow_volume(df: pd.DataFrame) -> pd.Series:
    cl = df['close'] - df['low']
    hc = df['high'] - df['close']
    hl = (df['high'] - df['low']).replace(0, np.nan)
    total_volume = _get_total_volume(df)
    mfv = (((cl - hc) / hl) * total_volume).fillna(0)
    return mfv


def _calculate_wma(data: pd.Series, period: int) -> pd.Series:
    """Calculates Weighted Moving Average using fast np.convolve."""
    if period < 1:
        return pd.Series(index=data.index, dtype=float)
    if period == 1:
        return data
    weights = np.arange(1, period + 1)
    weights_sum = weights.sum()
    wma_values = np.convolve(data.to_numpy(), weights, mode='valid') / weights_sum
    wma_full = np.full(len(data), np.nan)
    wma_full[period - 1:] = wma_values
    return pd.Series(wma_full, index=data.index, dtype=float)


def _calculate_rsi(data: pd.Series, period: int) -> pd.Series:
    if period <= 1:
        return pd.Series(50.0, index=data.index)
    delta = data.diff(1)
    gain = delta.clip(lower=0)
    loss = delta.clip(upper=0).abs()
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(100)
    rsi.loc[avg_gain == 0] = 50
    return rsi


def _calculate_rolling_sum_ratio(
        numerator_series: pd.Series,
        denominator_series: pd.Series,
        periods: Iterable[int],
        column_template: str
) -> pd.DataFrame:
    """
    Generic helper to calculate (rolling_sum(A) / rolling_sum(B)) for multiple periods.
    """
    series_list = []
    for period in periods:
        if period <= 0:
            continue
        sum_numerator = numerator_series.rolling(window=period).sum()
        sum_denominator = denominator_series.rolling(window=period).sum()
        column_name = column_template.format(period)
        final_series = (sum_numerator / sum_denominator.replace(0, np.nan)).rename(column_name)
        series_list.append(final_series)
    return pd.concat(series_list, axis=1)


def add_sma_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("SMA Calculations")
    sma_periods = range(5, 201)
    sma_series_list = []
    for period in sma_periods:
        column_name = f'sma_{period}'
        sma_series = df['close'].rolling(window=period).mean().rename(column_name)
        sma_series_list.append(sma_series)
    return pd.concat(sma_series_list, axis=1)


def add_ema_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("EMA Calculations")
    ema_periods = range(5, 201)
    ema_series_list = []
    for period in ema_periods:
        column_name = f'ema_{period}'
        ema_series = df['close'].ewm(span=period, adjust=False).mean().rename(column_name)
        ema_series_list.append(ema_series)
    return pd.concat(ema_series_list, axis=1)


def add_wma_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("WMA Calculations")
    wma_periods = range(5, 201)
    wma_series_list = []
    for period in wma_periods:
        column_name = f'wma_{period}'
        wma_series = _calculate_wma(df['close'], period).rename(column_name)
        wma_series_list.append(wma_series)
    return pd.concat(wma_series_list, axis=1)


def add_vwma_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("VWMA Calculations")
    price_x_volume = df['close'] * df['tick_count']
    volume = df['tick_count']
    return _calculate_rolling_sum_ratio(
        numerator_series=price_x_volume,
        denominator_series=volume,
        periods=range(5, 201),
        column_template="vwma_{}"
    )


def add_hma_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("HMA Calculations")
    hma_periods = range(5, 201)
    hma_series_list = []
    for period in hma_periods:
        column_name = f'hma_{period}'
        p_half = max(1, int(round(period / 2)))
        p_full = max(1, period)
        p_sqrt = max(1, int(round(math.sqrt(period))))
        wma_half = _calculate_wma(df['close'], p_half)
        wma_full = _calculate_wma(df['close'], p_full)
        diff_series = (2 * wma_half) - wma_full
        hma_series = _calculate_wma(diff_series, p_sqrt).rename(column_name)
        hma_series_list.append(hma_series)
    return pd.concat(hma_series_list, axis=1)


def add_smma_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("SMMA Calculations")
    smma_periods = range(5, 201)
    smma_series_list = []
    for period in smma_periods:
        if period == 0: continue
        column_name = f'smma_{period}'
        smma_series = df['close'].ewm(alpha=1 / period, adjust=False).mean().rename(column_name)
        smma_series_list.append(smma_series)
    return pd.concat(smma_series_list, axis=1)


def add_dema_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("DEMA Calculations")
    dema_periods = range(5, 201)
    dema_series_list = []
    for period in dema_periods:
        column_name = f'dema_{period}'
        ema1 = df['close'].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        dema_series = (2 * ema1 - ema2).rename(column_name)
        dema_series_list.append(dema_series)
    return pd.concat(dema_series_list, axis=1)


def add_tema_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("TEMA Calculations")
    tema_periods = range(5, 201)
    tema_series_list = []
    for period in tema_periods:
        column_name = f'tema_{period}'
        ema1 = df['close'].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        tema_series = (3 * ema1 - 3 * ema2 + ema3).rename(column_name)
        tema_series_list.append(tema_series)
    return pd.concat(tema_series_list, axis=1)


def add_rsi_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("RSI Calculations")
    rsi_periods = range(5, 201)
    rsi_series_list = []
    for period in rsi_periods:
        column_name = f'rsi_{period}'
        rsi_series = _calculate_rsi(df['close'], period).rename(column_name)
        rsi_series_list.append(rsi_series)
    return pd.concat(rsi_series_list, axis=1)


def add_cci_calculations(df: pd.DataFrame, periods: Iterable[int] = range(5, 201)) -> pd.DataFrame:
    """
    Compute CCI for multiple periods efficiently using Numba-accelerated rolling MAD.
    Returns a DataFrame with columns: cci_{period}
    """
    logging.info("CCI Calculations (Numba-accelerated)")
    tp = _get_typical_price(df)  # float32
    cci_series_list = []
    for period in periods:
        if period <= 1:
            continue
        col_name = f"cci_{period}"
        tp_sma = tp.rolling(window=period, min_periods=period).mean()
        # noinspection PyTypeChecker
        mad = tp.rolling(window=period, min_periods=period).apply(
            _mad_numba, raw=True, engine="numba", engine_kwargs={"parallel": True}
        )
        denom = 0.015 * mad.replace(0.0, np.nan)
        cci_series = (tp - tp_sma) / denom
        cci_series_list.append(cci_series.astype("float32").rename(col_name))
    return pd.concat(cci_series_list, axis=1)


def add_momentum_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Momentum Calculations")
    mom_periods = range(5, 201)
    mom_series_list = []
    for period in mom_periods:
        column_name = f'mom_{period}'
        mom_series = df['close'].diff(period).rename(column_name)
        mom_series_list.append(mom_series)
    return pd.concat(mom_series_list, axis=1)


def add_roc_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("ROC Calculations")
    roc_periods = range(5, 201)
    roc_series_list = []
    for period in roc_periods:
        column_name = f'roc_{period}'
        close_n = df['close'].shift(period)
        roc_series = ((df['close'] - close_n) / close_n.replace(0, np.nan)) * 100
        roc_series_list.append(roc_series.rename(column_name))
    return pd.concat(roc_series_list, axis=1)


def add_wpr_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("WPR Calculations")
    wpr_periods = range(5, 201)
    wpr_series_list = []
    for period in wpr_periods:
        column_name = f'wpr_{period}'
        high_n = df['high'].rolling(window=period).max()
        low_n = df['low'].rolling(window=period).min()
        range_n = (high_n - low_n).replace(0, np.nan)
        wpr_series = ((high_n - df['close']) / range_n) * -100
        wpr_series_list.append(wpr_series.rename(column_name))
    return pd.concat(wpr_series_list, axis=1)


def add_atr_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("ATR Calculations")
    atr_periods = range(5, 201)
    atr_series_list = []
    tr = _get_true_range(df)
    for period in atr_periods:
        if period == 0: continue
        column_name = f'atr_{period}'
        atr_series = tr.ewm(alpha=1 / period, adjust=False).mean().rename(column_name)
        atr_series_list.append(atr_series)
    return pd.concat(atr_series_list, axis=1)


def add_adx_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("ADX Calculations")
    adx_periods = range(5, 201)
    adx_series_list = []
    tr = _get_true_range(df)
    high_diff = df['high'].diff(1)
    low_diff = df['low'].diff(1) * -1
    high_diff_pos = high_diff.clip(lower=0)
    low_diff_pos = low_diff.clip(lower=0)
    plus_dm_arr = np.where(high_diff_pos > low_diff_pos, high_diff_pos, 0.0)
    minus_dm_arr = np.where(low_diff_pos > high_diff_pos, low_diff_pos, 0.0)
    plus_dm = pd.Series(plus_dm_arr, index=df.index)
    minus_dm = pd.Series(minus_dm_arr, index=df.index)
    for period in adx_periods:
        if period <= 1: continue
        column_name_adx = f'adx_{period}'
        column_name_pdi = f'pdi_{period}'
        column_name_mdi = f'mdi_{period}'
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        plus_dms = plus_dm.ewm(alpha=1 / period, adjust=False).mean()
        minus_dms = minus_dm.ewm(alpha=1 / period, adjust=False).mean()
        atr_safe = atr.replace(0, np.nan)
        pdi = (100 * (plus_dms / atr_safe)).rename(column_name_pdi)
        mdi = (100 * (minus_dms / atr_safe)).rename(column_name_mdi)
        di_sum = (pdi + mdi).replace(0, np.nan)
        di_diff = (pdi - mdi).abs()
        dx = (100 * (di_diff / di_sum))
        adx = dx.ewm(alpha=1 / period, adjust=False).mean().rename(column_name_adx)
        adx_series_list.append(adx)
        adx_series_list.append(pdi)
        adx_series_list.append(mdi)
    return pd.concat(adx_series_list, axis=1)


def add_trix_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("TRIX Calculations")
    trix_periods = range(5, 201)
    trix_series_list = []
    for period in trix_periods:
        column_name = f'trix_{period}'
        ema1 = df['close'].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        ema3_n = ema3.shift(1)
        trix = ((ema3 - ema3_n) / ema3_n.replace(0, np.nan)) * 100
        trix_series_list.append(trix.rename(column_name))
    return pd.concat(trix_series_list, axis=1)


def add_smoothed_rsi_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Smoothed RSI Calculations")
    rsx_periods = range(5, 201)
    rsx_series_list = []
    for period in rsx_periods:
        if period <= 1: continue
        column_name = f'sm_rsi_{period}'
        rsi = _calculate_rsi(df['close'], period)
        sm_rsi = rsi.ewm(alpha=1 / period, adjust=False).mean().rename(column_name)
        rsx_series_list.append(sm_rsi)
    return pd.concat(rsx_series_list, axis=1)


def add_donchian_channel_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Donchian Channel Calculations")
    dc_periods = range(5, 201)
    dc_series_list = []
    for period in dc_periods:
        upper = df['high'].rolling(window=period).max().rename(f'donchian_upper_{period}')
        lower = df['low'].rolling(window=period).min().rename(f'donchian_lower_{period}')
        middle = ((upper + lower) / 2).rename(f'donchian_middle_{period}')
        dc_series_list.append(upper)
        dc_series_list.append(lower)
        dc_series_list.append(middle)
    return pd.concat(dc_series_list, axis=1)


def add_envelope_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Envelope Calculations")
    env_periods = range(5, 201)
    env_series_list = []
    envelope_percent = 0.025
    for period in env_periods:
        middle = df['close'].rolling(window=period).mean().rename(f'env_middle_{period}')
        upper = (middle * (1 + envelope_percent)).rename(f'env_upper_{period}')
        lower = (middle * (1 - envelope_percent)).rename(f'env_lower_{period}')
        env_series_list.append(upper)
        env_series_list.append(lower)
        env_series_list.append(middle)
    return pd.concat(env_series_list, axis=1)


def add_fractal_chaos_bands_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Fractal Chaos Bands Calculations")
    fcb_periods = range(5, 201)
    fcb_series_list = []
    for period in fcb_periods:
        upper_raw = df['high'].rolling(window=period).max()
        lower_raw = df['low'].rolling(window=period).min()
        upper = upper_raw.shift(1).rename(f'fcb_upper_{period}')
        lower = lower_raw.shift(1).rename(f'fcb_lower_{period}')
        fcb_series_list.append(upper)
        fcb_series_list.append(lower)
    return pd.concat(fcb_series_list, axis=1)


def add_obv_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("OBV Calculations")
    total_volume = _get_total_volume(df)
    close_diff = df['close'].diff(1)
    obv_direction_sign = np.sign(close_diff)
    obv_direction = (obv_direction_sign * total_volume).fillna(0)
    obv_line = obv_direction.cumsum().rename('obv_line')
    obv_sma_list = [obv_line]
    sma_periods = range(5, 201)
    for period in sma_periods:
        column_name = f'obv_sma_{period}'
        obv_sma = obv_line.rolling(window=period).mean().rename(column_name)
        obv_sma_list.append(obv_sma)
    return pd.concat(obv_sma_list, axis=1)


def add_chaikin_money_flow_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Chaikin Money Flow Calculations")
    mfv = _get_money_flow_volume(df)
    total_volume = _get_total_volume(df)
    return _calculate_rolling_sum_ratio(
        numerator_series=mfv,
        denominator_series=total_volume,
        periods=range(5, 201),
        column_template="cmf_{}"
    )


def add_force_index_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Force Index Calculations")
    fi_periods = range(5, 201)
    fi_series_list = []
    total_volume = _get_total_volume(df)
    close_diff = df['close'].diff(1)
    fi_1 = (close_diff * total_volume).fillna(0)
    for period in fi_periods:
        column_name = f'fi_{period}'
        fi_n_series = fi_1.ewm(span=period, adjust=False).mean().rename(column_name)
        fi_series_list.append(fi_n_series)
    return pd.concat(fi_series_list, axis=1)


def add_money_flow_index_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Money Flow Index Calculations")
    mfi_periods = range(5, 201)
    mfi_series_list = []
    tp = _get_typical_price(df)
    total_volume = _get_total_volume(df)
    raw_money_flow = tp * total_volume
    tp_diff = tp.diff(1)
    tp_sign = np.sign(tp_diff)
    pos_mf = (raw_money_flow * (tp_sign > 0)).fillna(0)
    neg_mf = (raw_money_flow * (tp_sign < 0)).fillna(0)
    for period in mfi_periods:
        if period <= 1: continue
        column_name = f'mfi_{period}'
        sum_pos_mf = pos_mf.rolling(window=period).sum()
        sum_neg_mf = neg_mf.rolling(window=period).sum()
        mfr = sum_pos_mf / sum_neg_mf.replace(0, np.nan)
        mfi = 100.0 - (100.0 / (1.0 + mfr))
        mfi = mfi.fillna(100)
        mfi.loc[sum_pos_mf == 0] = 50
        mfi_series_list.append(mfi.rename(column_name))
    return pd.concat(mfi_series_list, axis=1)


def add_vortex_indicator_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Vortex Indicator Calculations")
    vi_periods = range(5, 201)
    vi_series_list = []
    tr = _get_true_range(df)
    vm_plus = (df['high'] - df['low'].shift(1)).abs()
    vm_minus = (df['low'] - df['high'].shift(1)).abs()
    for period in vi_periods:
        if period == 0: continue
        sum_tr = tr.rolling(window=period).sum().replace(0, np.nan)
        sum_vm_plus = vm_plus.rolling(window=period).sum()
        sum_vm_minus = vm_minus.rolling(window=period).sum()
        vi_plus = (sum_vm_plus / sum_tr).rename(f'vi_plus_{period}')
        vi_minus = (sum_vm_minus / sum_tr).rename(f'vi_minus_{period}')
        vi_series_list.append(vi_plus)
        vi_series_list.append(vi_minus)
    return pd.concat(vi_series_list, axis=1)


def add_aroon_indicator_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Aroon Indicator Calculations (Numba deque + parallel)")
    highs = df["high"].to_numpy(dtype=np.float32, copy=False)
    lows = df["low"].to_numpy(dtype=np.float32, copy=False)
    periods_arr = np.arange(5, 201, dtype=np.int32)
    up_all, down_all = _compute_aroon_all(highs, lows, periods_arr)
    all_cols = {}
    for j, p in enumerate(periods_arr):
        up_col = f"aroon_up_{p}"
        down_col = f"aroon_down_{p}"
        osc_col = f"aroon_osc_{p}"
        up_data = up_all[:, j]
        down_data = down_all[:, j]
        osc_data = up_data - down_data
        all_cols[up_col] = up_data
        all_cols[down_col] = down_data
        all_cols[osc_col] = osc_data.astype(np.float32)

    return pd.DataFrame(all_cols, index=df.index)


def add_chande_momentum_oscillator_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Chande Momentum Oscillator Calculations")
    cmo_periods = range(5, 201)
    cmo_series_list = []
    delta = df['close'].diff(1)
    gain = delta.clip(lower=0)
    loss = delta.clip(upper=0).abs()
    for period in cmo_periods:
        if period == 0: continue
        column_name = f'cmo_{period}'
        sum_gain = gain.rolling(window=period).sum()
        sum_loss = loss.rolling(window=period).sum()
        sum_total = (sum_gain + sum_loss).replace(0, np.nan)
        cmo = (100 * (sum_gain - sum_loss) / sum_total).rename(column_name)
        cmo_series_list.append(cmo)
    return pd.concat(cmo_series_list, axis=1)


def add_standard_deviation_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Standard Deviation Calculations")
    std_periods = range(5, 201)
    std_series_list = []
    for period in std_periods:
        column_name = f'std_dev_{period}'
        std_series = df['close'].rolling(window=period).std().rename(column_name)
        std_series_list.append(std_series)
    return pd.concat(std_series_list, axis=1)


def add_variance_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Variance Calculations")
    var_periods = range(5, 201)
    var_series_list = []
    for period in var_periods:
        column_name = f'variance_{period}'
        var_series = df['close'].rolling(window=period).var().rename(column_name)
        var_series_list.append(var_series)
    return pd.concat(var_series_list, axis=1)


def add_median_filter_calculations(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Median Filter Calculations (Numba-accelerated)")
    med_periods = range(5, 201)
    med_series_list = []
    close_series = df['close'].astype("float32")
    for period in med_periods:
        column_name = f'median_{period}'
        # noinspection PyTypeChecker
        med_series = close_series.rolling(window=period, min_periods=period).apply(
            _get_median, raw=True, engine="numba", engine_kwargs={"parallel": True}
        ).rename(column_name)
        med_series_list.append(med_series)
    return pd.concat(med_series_list, axis=1)