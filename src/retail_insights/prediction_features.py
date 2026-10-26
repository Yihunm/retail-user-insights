"""固定 90 日历史窗口的时间特征与具有完整观察期的复购/流失标签。"""

import re
from datetime import date, datetime, timedelta
from numbers import Integral

import numpy as np
import pandas as pd

FEATURES = ['recency_days', 'orders_7', 'orders_30', 'orders_90', 'revenue_7', 'revenue_30',
            'revenue_90', 'quantity_90', 'unique_products_90', 'active_days_90',
            'average_order_value_90', 'max_order_value_90', 'interval_mean_days',
            'interval_missing', 'history_span_days']
FEATURE_VERSION = '1'
DEFAULT_CUTOFFS = ['2011-03-02', '2011-04-01', '2011-05-01', '2011-06-01', '2011-08-01', '2011-10-01']


def _date(value, name='cutoff'):
    if isinstance(value, str):
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', value) is None:
            raise ValueError(f'{name} 日期必须是完整 YYYY-MM-DD 或无时区日期对象')
    elif not isinstance(value, (date, datetime, pd.Timestamp)):
        raise ValueError(f'{name} 日期必须是完整 YYYY-MM-DD 或无时区日期对象')
    try:
        result = pd.Timestamp(value)
        if pd.isna(result) or result.tzinfo is not None:
            raise ValueError
        return result.as_unit('ns').normalize()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} 日期无效或包含时区') from exc


def _window(window_days):
    if isinstance(window_days, bool) or not isinstance(window_days, Integral) or window_days != 90:
        raise ValueError('历史窗口 window_days 固定为整数 90 天')


def _shift(timestamp, days):
    try:
        return timestamp + timedelta(days=days)
    except (ValueError, OverflowError) as exc:
        raise ValueError('日期或预测天数超出可支持范围') from exc


def _coverage(sales):
    if sales.empty:
        raise ValueError('销售交易数据为空，无法确定历史及标签覆盖范围')
    required = ['CustomerID', 'InvoiceNo', 'InvoiceDate', 'StockCode', 'Quantity', 'AmountCents']
    missing = [column for column in required if column not in sales]
    if missing:
        raise ValueError('销售数据缺少列：' + ', '.join(missing))
    dates = sales.InvoiceDate
    if not pd.api.types.is_datetime64_any_dtype(dates.dtype) or isinstance(dates.dtype, pd.DatetimeTZDtype) or dates.isna().any():
        raise ValueError('销售 InvoiceDate 必须是清洗后的有效无时区日期')
    return dates.min().ceil('D'), dates.max().floor('D')


def _known_customers(values):
    return values.notna() & values.astype('string').str.strip().ne('')


def _sum(values):
    return sum(value.item() if isinstance(value, np.generic) else value for value in values)


def _empty_features():
    return pd.DataFrame({'CustomerID': pd.Series(dtype='string'),
                         **{feature: pd.Series(dtype='float64') for feature in FEATURES}})


def feature_frame(sales, cutoff, window_days=90):
    """仅用 [T-90,T) 的已知客户销售计算固定顺序的 15 个数值特征。

    金额在 Python 整数分中汇总后转换为英镑浮点；Quantity 先 Python 标量求和。
    InvoiceNo 区分订单；订单间隔使用各订单首条交易时间，只有一单时为 NaN。
    开头不完整日不算可用历史，T 不能超过数据最后销售日的次日零点。
    """
    _window(window_days)
    timestamp = _date(cutoff)
    coverage_start, last_day = _coverage(sales)
    lower = _shift(timestamp, -90)
    if lower < coverage_start:
        raise ValueError('cutoff 截点之前没有完整 90 天历史覆盖')
    if timestamp > _shift(last_day, 1):
        raise ValueError('cutoff 截点超出销售数据覆盖范围，不能使用未来截点')
    mask = _known_customers(sales.CustomerID) & sales.InvoiceDate.ge(lower) & sales.InvoiceDate.lt(timestamp)
    selected = sales.loc[mask].copy()
    selected['CustomerID'] = selected.CustomerID.astype('string')
    records = []
    for customer, rows in selected.groupby('CustomerID', sort=True):
        first_times = rows.groupby('InvoiceNo', sort=False).InvoiceDate.min().sort_values()
        orders = len(first_times)
        cents = sum(int(value) for value in rows.AmountCents)
        order_cents = [sum(int(value) for value in group.AmountCents)
                       for _, group in rows.groupby('InvoiceNo', sort=False)]
        latest, earliest = rows.InvoiceDate.max(), rows.InvoiceDate.min()
        record = {
            'CustomerID': customer,
            'recency_days': (timestamp - latest.normalize()).days,
            'orders_90': orders, 'revenue_90': cents / 100,
            'quantity_90': _sum(rows.Quantity),
            'unique_products_90': rows.StockCode.nunique(),
            'active_days_90': rows.InvoiceDate.dt.normalize().nunique(),
            'average_order_value_90': cents / 100 / orders,
            'max_order_value_90': max(order_cents) / 100,
            'interval_mean_days': first_times.diff().dt.total_seconds().mean() / 86400 if orders > 1 else np.nan,
            'interval_missing': int(orders == 1),
            'history_span_days': (latest.normalize() - earliest.normalize()).days,
        }
        for days in [7, 30]:
            recent = rows.loc[rows.InvoiceDate.ge(_shift(timestamp, -days))]
            record[f'orders_{days}'] = recent.InvoiceNo.nunique()
            record[f'revenue_{days}'] = sum(int(value) for value in recent.AmountCents) / 100
        records.append(record)
    result = pd.DataFrame(records, columns=['CustomerID', *FEATURES]) if records else _empty_features()
    result['CustomerID'] = result.CustomerID.astype('string')
    # Model features intentionally use floats, after exact monetary/integer aggregation.
    result[FEATURES] = result[FEATURES].astype(float)
    result.attrs.update({'cutoff': timestamp.date().isoformat(), 'window_days': 90,
                         'features': FEATURES.copy(), 'feature_version': FEATURE_VERSION})
    return result


def make_samples(sales, cutoffs=None, task='purchase', horizon_days=None, window_days=90, coverage_end=None):
    """构造完整历史和未来观察期的客户样本，绝不把未知未来当作负标签。

    purchase: H 默认 30 日，未来 [T,T+H) 至少购买一次为 1；
    churn: H 默认 60 日，未来 [T,T+H) 没有购买为 1。
    数据最后交易所在日视为不完整日；即使手动指定 coverage_end 也不能越过它。
    """
    _window(window_days)
    if task not in ['purchase', 'churn']:
        raise ValueError('任务 task 只能是 purchase 或 churn')
    if horizon_days is None:
        horizon_days = 30 if task == 'purchase' else 60
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, Integral) or horizon_days <= 0:
        raise ValueError('预测标签 horizon_days 必须是正整数天数')
    horizon_days = int(horizon_days)
    coverage_start, observed_end = _coverage(sales)
    complete_end = _date(coverage_end, 'coverage_end') if coverage_end is not None else observed_end
    if complete_end > observed_end or complete_end < coverage_start:
        raise ValueError('coverage_end 必须位于已观测完整日期覆盖范围内')
    if cutoffs is None:
        cutoffs = DEFAULT_CUTOFFS
    if isinstance(cutoffs, (str, bytes)):
        raise ValueError('cutoffs 必须是日期列表')
    try:
        timestamps = [_date(value) for value in cutoffs]
    except TypeError as exc:
        raise ValueError('cutoffs 必须是日期列表') from exc
    if len(timestamps) != len(set(timestamps)):
        raise ValueError('cutoffs 含有重复截点日期')
    frames = []
    for timestamp in timestamps:
        label_end = _shift(timestamp, horizon_days)
        if label_end > complete_end:
            raise ValueError(f'{timestamp.date().isoformat()} 的未来标签观察期没有完整覆盖')
        features = feature_frame(sales, timestamp, window_days=90)
        future = sales.loc[sales.InvoiceDate.ge(timestamp) & sales.InvoiceDate.lt(label_end)]
        purchased = set(future.loc[_known_customers(future.CustomerID), 'CustomerID'].astype('string'))
        features.insert(1, 'Cutoff', timestamp.date().isoformat())
        features.insert(2, 'Task', task)
        targets = features.CustomerID.isin(purchased).astype('int64')
        features['Target'] = targets if task == 'purchase' else 1 - targets
        # Per-cutoff metadata is replaced with a complete, shared sample contract.
        features.attrs = {}
        frames.append(features)
    if frames:
        result = pd.concat(frames, ignore_index=True)
    else:
        result = _empty_features()
        result.insert(1, 'Cutoff', pd.Series(dtype='string'))
        result.insert(2, 'Task', pd.Series(dtype='string'))
        result['Target'] = pd.Series(dtype='int64')
    result.attrs.update({
        'task': task, 'horizon_days': horizon_days, 'window_days': 90,
        'coverage_start': coverage_start.date().isoformat(), 'coverage_end': complete_end.date().isoformat(),
        'features': FEATURES.copy(), 'feature_version': FEATURE_VERSION,
        'cutoffs': [timestamp.date().isoformat() for timestamp in timestamps],
    })
    return result
