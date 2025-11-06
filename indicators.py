# indicators.py

import pandas as pd
import numpy as np
import math


def _get_typical_price(df: pd.DataFrame) -> pd.Series:
    return (df['high'] + df['low'] + df['close']) / 3


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
    if period < 1:
        return pd.Series(index=data.index, dtype=float)
    if period == 1:
        return data
    weights = np.arange(1, period + 1)
    def wma_calc(window_data):
        return np.dot(window_data, weights) / weights.sum()
    return data.rolling(window=period).apply(wma_calc, raw=True)


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


def add_sma_calculations(df: pd.DataFrame) -> pd.DataFrame:
    sma_periods = range(5, 201)
    sma_series_list = []
    for period in sma_periods:
        column_name = f'sma_{period}'
        sma_series = df['close'].rolling(window=period).mean().rename(column_name)
        sma_series_list.append(sma_series)
    return pd.concat(sma_series_list, axis=1)


def add_ema_calculations(df: pd.DataFrame) -> pd.DataFrame:
    ema_periods = range(5, 201)
    ema_series_list = []
    for period in ema_periods:
        column_name = f'ema_{period}'
        ema_series = df['close'].ewm(span=period, adjust=False).mean().rename(column_name)
        ema_series_list.append(ema_series)
    return pd.concat(ema_series_list, axis=1)


def add_wma_calculations(df: pd.DataFrame) -> pd.DataFrame:
    wma_periods = range(5, 201)
    wma_series_list = []
    for period in wma_periods:
        column_name = f'wma_{period}'
        wma_series = _calculate_wma(df['close'], period).rename(column_name)
        wma_series_list.append(wma_series)
    return pd.concat(wma_series_list, axis=1)


def add_vwma_calculations(df: pd.DataFrame) -> pd.DataFrame:
    vwma_periods = range(5, 201)
    vwma_series_list = []
    price_x_volume = df['close'] * df['tick_count']
    volume = df['tick_count']
    for period in vwma_periods:
        column_name = f'vwma_{period}'
        sum_price_x_volume = price_x_volume.rolling(window=period).sum()
        sum_volume = volume.rolling(window=period).sum()
        vwma_series = (sum_price_x_volume / sum_volume.replace(0, np.nan)).rename(column_name)
        vwma_series_list.append(vwma_series)
    return pd.concat(vwma_series_list, axis=1)


def add_hma_calculations(df: pd.DataFrame) -> pd.DataFrame:
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
    smma_periods = range(5, 201)
    smma_series_list = []
    for period in smma_periods:
        if period == 0: continue
        column_name = f'smma_{period}'
        smma_series = df['close'].ewm(alpha=1 / period, adjust=False).mean().rename(column_name)
        smma_series_list.append(smma_series)
    return pd.concat(smma_series_list, axis=1)


def add_dema_calculations(df: pd.DataFrame) -> pd.DataFrame:
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
    rsi_periods = range(5, 201)
    rsi_series_list = []
    for period in rsi_periods:
        column_name = f'rsi_{period}'
        rsi_series = _calculate_rsi(df['close'], period).rename(column_name)
        rsi_series_list.append(rsi_series)
    return pd.concat(rsi_series_list, axis=1)


def add_cci_calculations(df: pd.DataFrame) -> pd.DataFrame:
    cci_periods = range(5, 201)
    cci_series_list = []
    tp = _get_typical_price(df)
    for period in cci_periods:
        if period <= 1: continue
        column_name = f'cci_{period}'
        tp_sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: (x - x.mean()).abs().mean(), raw=False)
        cci_series = (tp - tp_sma) / (0.015 * mad.replace(0, np.nan))
        cci_series_list.append(cci_series.rename(column_name))
    return pd.concat(cci_series_list, axis=1)


def add_momentum_calculations(df: pd.DataFrame) -> pd.DataFrame:
    mom_periods = range(5, 201)
    mom_series_list = []
    for period in mom_periods:
        column_name = f'mom_{period}'
        mom_series = df['close'].diff(period).rename(column_name)
        mom_series_list.append(mom_series)
    return pd.concat(mom_series_list, axis=1)


def add_roc_calculations(df: pd.DataFrame) -> pd.DataFrame:
    roc_periods = range(5, 201)
    roc_series_list = []
    for period in roc_periods:
        column_name = f'roc_{period}'
        close_n = df['close'].shift(period)
        roc_series = ((df['close'] - close_n) / close_n.replace(0, np.nan)) * 100
        roc_series_list.append(roc_series.rename(column_name))
    return pd.concat(roc_series_list, axis=1)


def add_wpr_calculations(df: pd.DataFrame) -> pd.DataFrame:
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
    cmf_periods = range(5, 201)
    cmf_series_list = []
    mfv = _get_money_flow_volume(df)
    total_volume = _get_total_volume(df)
    for period in cmf_periods:
        column_name = f'cmf_{period}'
        sum_mfv = mfv.rolling(window=period).sum()
        sum_vol = total_volume.rolling(window=period).sum()
        cmf_series = (sum_mfv / sum_vol.replace(0, np.nan)).rename(column_name)
        cmf_series_list.append(cmf_series)
    return pd.concat(cmf_series_list, axis=1)


def add_force_index_calculations(df: pd.DataFrame) -> pd.DataFrame:
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
    aroon_periods = range(5, 201)
    aroon_series_list = []
    for period in aroon_periods:
        if period <= 1: continue
        column_up = f'aroon_up_{period}'
        column_down = f'aroon_down_{period}'
        column_osc = f'aroon_osc_{period}'
        periods_since_high_series = df['high'].rolling(window=period).apply(
            lambda x: (period - 1) - np.argmax(x), raw=True
        )
        periods_since_low_series = df['low'].rolling(window=period).apply(
            lambda x: (period - 1) - np.argmin(x), raw=True
        )
        divisor = float(period - 1)
        aroon_up = (((divisor - periods_since_high_series) / divisor) * 100).rename(column_up)
        aroon_down = (((divisor - periods_since_low_series) / divisor) * 100).rename(column_down)
        aroon_osc = (aroon_up - aroon_down).rename(column_osc)
        aroon_series_list.append(aroon_up)
        aroon_series_list.append(aroon_down)
        aroon_series_list.append(aroon_osc)
    return pd.concat(aroon_series_list, axis=1)


def add_chande_momentum_oscillator_calculations(df: pd.DataFrame) -> pd.DataFrame:
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
    std_periods = range(5, 201)
    std_series_list = []
    for period in std_periods:
        column_name = f'std_dev_{period}'
        std_series = df['close'].rolling(window=period).std().rename(column_name)
        std_series_list.append(std_series)
    return pd.concat(std_series_list, axis=1)


def add_variance_calculations(df: pd.DataFrame) -> pd.DataFrame:
    var_periods = range(5, 201)
    var_series_list = []
    for period in var_periods:
        column_name = f'variance_{period}'
        var_series = df['close'].rolling(window=period).var().rename(column_name)
        var_series_list.append(var_series)
    return pd.concat(var_series_list, axis=1)


def add_median_filter_calculations(df: pd.DataFrame) -> pd.DataFrame:
    med_periods = range(5, 201)
    med_series_list = []
    for period in med_periods:
        column_name = f'median_{period}'
        med_series = df['close'].rolling(window=period).median().rename(column_name)
        med_series_list.append(med_series)
    return pd.concat(med_series_list, axis=1)